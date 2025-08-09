from transformers import TrainingArguments, Trainer
from models.modelbuilder import LLMModelBuilder
from utils.configreader import ConfigReader
import time


class StartTraining(LLMModelBuilder):
    def __init__(self, tokenized_datasets, cls):
        super().__init__()
        self.tokenized_datasets = tokenized_datasets
        self.model = model
        self.config_data = ConfigReader().get_yaml_data()
        self.output_dir = self.config_data['training_param']['checkpoints_path'] + (str(int(time.time())))
        self.lr = float(self.config_data['training_param']['learning_rate'])
        self.epoch = int(self.config_data['training_param']['num_train_epoch'])
        self.wgt_decay = int(self.config_data['training_param']['weight_decay'])
        self.log_steps = int(self.config_data['training_param']['logging_steps'])
        self.max_steps = int(self.config_data['training_param']['max_steps'])

    def go(self):
        print(f'Checkpoints path would be : {self.output_dir}')
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            learning_rate=self.lr,
            num_train_epochs=self.epoch,
            weight_decay=self.wgt_decay,
            logging_steps=self.log_steps,
            max_steps=self.max_steps
        )
        print(f'cls : {self.model}')

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_datasets['train'],
            eval_dataset=self.tokenized_datasets['validation']
        )

        print(f'Training started .....')

        trainer.train()