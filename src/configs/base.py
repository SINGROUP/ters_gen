import yaml
from ml_collections import config_dict


def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def get_config():
    cfg = config_dict.ConfigDict()
    cfg_dataloader = load_yaml_config('src/configs/dataloader.yaml')
    cfg_dataset = load_yaml_config('src/configs/dataset.yaml')
    cfg_model = load_yaml_config('src/configs/model.yaml')


    cfg.dataloader = config_dict.ConfigDict(cfg_dataloader)
    cfg.dataset = config_dict.ConfigDict(cfg_dataset)
    cfg.model = config_dict.ConfigDict(cfg_model)

    return cfg