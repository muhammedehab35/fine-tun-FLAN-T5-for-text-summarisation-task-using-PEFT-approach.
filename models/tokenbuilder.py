from transformers import AutoTokenizer
from models.modelbuilder import LLMModelBuilder
from datasets import dataset_dict


class TokenizerBuilder(LLMModelBuilder):
    def __init__(self):
        super().__init__()
        self.remove_columns = ['id', 'dialogue', 'summary']

    def                          get_tokenizer(self):
        print(f'Loading {self.model_name} tokenizer')
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        return tokenizer

    def pre_proc_tokenizer(self, input_example: dataset_dict.DatasetDict) -> dataset_dict.DatasetDict:
        """
        A pre-processing part of tokenising inputs
        :param input_example:
        :return: pre-processed datasets
        """
        # get tokenizer
        tokenizer = self.get_tokenizer()
        start_prompt = 'Summarize the following conversation.\n\n'
        end_prompt = '\n\nSummary:'
        prompt = [start_prompt + dialogue + end_prompt for dialogue in input_example['dialogue']]
        input_example['input_ids'] = tokenizer(prompt, padding='max_length', truncation=True,
                                               return_tensors='pt').input_ids.to(self.device)
        input_example['labels'] = tokenizer(input_example['summary'], padding='max_length', truncation=True,
                                            return_tensors='pt').input_ids.to(self.device)

        return input_example

    def get_tokenized_inputs(self, dataset: dataset_dict.DatasetDict):
        """
        This function takes care of tokenizing w.r.t model and converts all inputs to selected tokenizer
        :param dataset: datasets type contains input datasets
        :return: tokenized datasets
        """
        print(f'Tokenizing inputs datasets')
        tokenized_datasets = dataset.map(self.pre_proc_tokenizer, batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(self.remove_columns)

        return tokenized_datasets
