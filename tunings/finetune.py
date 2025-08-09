from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer
import time

from utils.configreader import ConfigReader
from utils.helpers import Helpers


class FineTune:
    def __init__(self, model, tokenized_datasets):
        self.model = model
        self.tokenized_datasets = tokenized_datasets
        self.config_data = ConfigReader().get_yaml_data()

        self.lora_param = self.config_data['finetune_param']['lora_config']
        self.peft_param = self.config_data['finetune_param']['peft_config']
        self.rank = self.lora_param['lora_rank']
        self.lora_alpha = self.lora_param['lora_alpha']
        self.target_modules = self.lora_param['target_modules']
        self.dropout = self.lora_param['dropout']
        self.bias = self.lora_param['bias']
        self.output_path = self.lora_param['checkpoints_path'] + str(int(time.time()))
        # peft configuration
        self.auto_fine_batch_size = bool(self.peft_param['auto_fine_batch_size'])
        self.lr = float(self.peft_param['learning_rate'])
        self.num_train_epochs = self.peft_param['num_train_epochs']
        self.logging_steps = self.peft_param['logging_steps']
        self.max_steps = self.peft_param['max_steps']

        self.lora_config = LoraConfig(
            r=self.rank,  # Rank
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.dropout,
            bias=self.bias,
            task_type=TaskType.SEQ_2_SEQ_LM  # FLAN-T5
        )

    def tune(self):
        peft_model = get_peft_model(self.model, self.lora_config)
        # printing number of trainable parameters after `LoRA`
        print('--' * 50)
        print(f'After applying LoRA fine tuning strategies, number of trainable parameters are :\n')
        print(Helpers.desc_number_of_trainable_parameters(peft_model))
        print('--' * 50)

        peft_training_args = TrainingArguments(
            output_dir=self.output_path,
            auto_find_batch_size=self.auto_fine_batch_size,
            learning_rate=self.lr,  # Higher learning rate than full fine-tuning.
            num_train_epochs=self.num_train_epochs,
            logging_steps=self.logging_steps,
            max_steps=self.max_steps
        )

        peft_trainer = Trainer(
            model=peft_model,
            args=peft_training_args,
            train_dataset=self.tokenized_datasets["train"],
        )

        print(f'Training Starting ...')
        peft_trainer.train()

        # saving configurations and params
        # peft_model_path = "./checkpoints/peft-dialogue-summary-checkpoint-local"
        # peft_trainer.model.save_pretrained(peft_model_path)
        # tokenizer.save_pretrained(peft_model_path)
