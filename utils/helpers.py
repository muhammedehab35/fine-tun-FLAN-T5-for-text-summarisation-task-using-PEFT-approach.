

class Helpers:
    def __init__(self):
        pass

    @staticmethod
    def desc_number_of_trainable_parameters(model):
        """
        This function computes the trainable parameters required to train, for pruning or fine-tuning strategies it
        will print only trainable parameters
        :param model: LLM model
        :return: string which consists of information about trainable parameters
        """
        trainable_model_params = 0
        all_model_params = 0
        for _, param in model.named_parameters():
            all_model_params += param.numel()
            if param.requires_grad:
                trainable_model_params += param.numel()
        # print(f'Trainable cls parameters : {trainable_model_params} \nAll cls parameters : {all_model_params}')
        return f"trainable cls parameters: {trainable_model_params}\nall cls parameters: " \
               f"{all_model_params}\npercentage of trainable cls parameters: " \
               f"{100 * trainable_model_params / all_model_params:.2f}%"


