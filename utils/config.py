import yaml
from pathlib import Path

global cfg
if 'cfg' not in globals():
    config_path = Path(__file__).resolve().parent / 'config.yml'
    with open(config_path, 'r') as file:
        cfg = yaml.safe_load(file)

if __name__ == '__main__':
    print(cfg)