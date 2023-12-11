# cs236-final-proj

Intructions to run:

1. Create a ws_config.json file with relevant information for your Azure ML Studio workspace
   ex.
   {
    "subscription_id": "XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX",
    "resource_group_name": "rg_name",
    "workspace_name": "ws_name"
   }
3. Run `python aml_submit.py` to submit code to cluster for training
