import yaml
from ml_collections import config_dict


def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def get_config(config_path):

    if config_path:
        cfg_yaml = load_yaml_config(config_path)
    else:
        cfg_yaml = load_yaml_config('src/configs/dataloader.yaml')

    
    
    cfg = config_dict.ConfigDict(cfg_yaml)

    return cfg