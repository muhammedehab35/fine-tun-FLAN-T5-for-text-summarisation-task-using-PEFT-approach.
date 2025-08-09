from transformers import AutoModelForSeq2SeqLM
from utils.configreader import ConfigReader


class LLMModelBuilder:
    def __init__(self, model_name='google/flan-t5-base'):
        self.model_name = model_name
        self.config_data = ConfigReader().get_yaml_data()
        self.device = self.config_data['device']

    def get_flan_model(self):
        """
        Get 'FLAN' family model
        :return: model
        """
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)

        return model
