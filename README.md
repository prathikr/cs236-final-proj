# cs236-final-proj

Intructions to run:

1. Create a ws_config.json file with relevant information for your Azure ML Studio workspace
   ex. <br>
   { <br>
    "subscription_id": "XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX", <br>
    "resource_group_name": "rg_name", <br>
    "workspace_name": "ws_name" <br>
   }
3. Run `python aml_submit.py` to submit code to cluster for training
