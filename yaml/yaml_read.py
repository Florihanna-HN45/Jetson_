#ham doc file yaml

import yaml

with open("global_config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
    #ham doc file: yaml.safe_load();
    #thay vi yaml.load()

print(config["system"]["name"])   #output: RobotVision


#ghi file: 
data_to_save = {
    "status": "ready",
    "threshold": 0.85
}

with open("output.yaml", "w", encoding="utf-8") as f:
    yaml.safe_dump(data_to_save, f, default_flow_style=False)

#ham yaml.safe_dump();
#state: "r", "w"