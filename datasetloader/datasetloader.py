from datasets import load_dataset


class DatasetLoader:
    def __init__(self, dataset_name='samsum'):
        self.dataset_name = dataset_name

    def get_dataset(self):
        print(f'Going to load dataset : {self.dataset_name}')
        dataset = load_dataset(self.dataset_name)
        print(f'Summary of loaded dataset :\n{dataset}')
        return dataset
