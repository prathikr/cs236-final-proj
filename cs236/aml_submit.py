import argparse
from pathlib import Path
import json

from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment, BuildContext 
from azure.identity import AzureCliCredential

# run test on automode workspace
ws_config = json.load(open("ws_config.json"))
subscription_id = ws_config["subscription_id"]
resource_group = ws_config["resource_group_name"]
workspace_name = ws_config["workspace_name"]
compute = "v100" # F4s-v2 or v100 or v100-2

def get_args(raw_args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_name", default="eeg-experiment", help="Experiment name for AML Workspace")

    args = parser.parse_args(raw_args)
    return args

def main(raw_args=None):
    args = get_args(raw_args)

    ml_client = MLClient(
        AzureCliCredential(), subscription_id, resource_group, workspace_name
    )

    root_dir = Path(__file__).resolve().parent
    environment_dir = root_dir / "environment"
    code_dir = root_dir / "DreamDiffusion"

    pytorch_job = command(
        code=code_dir,  # local path where the code is stored
        command="torchrun code/stageA1_eeg_pretrain.py --num_epoch 500",
        # command="torchrun --nproc_per_node=2 code/eeg_ldm.py --dataset EEG --num_epoch 50 --batch_size 2 --pretrain_mbm_path pretrains/eeg_pretrains/checkpoint.pth",
        environment="mangoml-acpt@latest",
        experiment_name=args.experiment_name,
        compute=compute,
        display_name="stageA1_eeg_pretrain_500epochs",
    )

    print("submitting PyTorch job for stageA1 eeg pretrain")
    pytorch_returned_job = ml_client.create_or_update(pytorch_job)
    print("submitted job")

    pytorch_aml_url = pytorch_returned_job.studio_url
    print("job link:", pytorch_aml_url)

if __name__ == "__main__":
    main()