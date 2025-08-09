import yaml
import os


class ConfigReader:
    def __init__(self):
        self.filename = os.path.join('configs', 'config')
        self.output = None

    def get_yaml_data(self):
        if self.output is None:
            with open(f'{self.filename}.yaml', 'r') as f:
                self.output = yaml.safe_load(f)
        return self.output