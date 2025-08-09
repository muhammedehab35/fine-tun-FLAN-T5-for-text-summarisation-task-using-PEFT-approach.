from datasetloader.datasetloader import DatasetLoader

from models.modelbuilder import LLMModelBuilder
from models.tokenbuilder import TokenizerBuilder
from models.training import StartTraining
from tunings.finetune import FineTune


def train_llm():
    # Get dataset
    dt_loader = DatasetLoader()
    dataset = dt_loader.get_dataset()

    # Get Model and respective tokenizer
    md_bl = LLMModelBuilder()
    md_tk = TokenizerBuilder()

    original_model = md_bl.get_flan_model()
    tokenizer = md_tk.get_tokenizer()

    tokenized_datasets = md_tk.get_tokenized_inputs(dataset=dataset)

    print(f'after tokenized datasets \n: {tokenized_datasets}')

    # for full fine-tuning
    # st_train = StartTraining(tokenized_datasets, original_model)
    # st_train.go()

    # for PEFT fine-tuning
    fnt = FineTune(model=original_model, tokenized_datasets=tokenized_datasets)
    fnt.tune()








if __name__ == '__main__':
    train_llm()