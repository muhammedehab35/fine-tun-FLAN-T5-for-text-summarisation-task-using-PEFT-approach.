# fine-tuning-FLAN-T5-for-text-summarisation-task-using-PEFT-approach.

This project is based on fine-tuning LLM models (FLAN-T5) for text summarisation task using PEFT approach.

Following dataset (huggingface dataset) is being used for fine-tuning
- [SAMSum](https://huggingface.co/datasets/knkarthick/samsum) (It contains about `16k` messenger-like conversations with summaries)
- [DialogSum](https://huggingface.co/datasets/knkarthick/dialogsum)  (contains `10,000+` dialogues with the corresponding manually labeled summaries and topics)

Fine tuning is being carried out with Google's `FLAN T5` and it's tokenizer. This project also includes following proposals to carry out fine-tuning:
- Prompt engineering
- Instruction based fine-tuning
- Full fine-tuning (would better to have huge datasets)
- PEFT (LoRA) fine-tuning

> Evaluation

The [ROUGE metric](https://en.wikipedia.org/wiki/ROUGE_(metric)) helps quantify the validity of summarizations produced by models. This metric evaluates not to perfection, but indicate overall increase in summarizations effectiveness.

Below are the stats on `ROUGE` score being achieved with fine-tuning on FLAN-T5 model on `samsum` datasets
A
|Model|rouge1|rouge2|rougeL|rougeLsum|
|-----|------|------|------|---------|
|FLAN T5 (base model)|0.2334158581572823|0.07603964187010573|0.20145520923859048|0.20145899339006135|
|PEFT (LoRA fine tuned)|0.40810631575616746|0.1633255794568712|0.32507074586565354|0.3248950182867091|



> Interpretation of improvement on PEFT fine-tuned models:

Absolute percentage improvement of `PEFT` MODEL over BASE model

|score matrix|% improvement|
|------------|-------------|
|rouge1| 17.47%|
|rouge2| 8.73%|
|rougeL| 12.36%|
|rougeLsum| 12.34%|


> Future Work

- [ ] Inclusion of inference code (though basic notebook exists for inference)
- [ ] Adaptation of this framework with custom data-loader for pre-defined format
- [ ] Include other strategies for `PEFT` computation
- [ ] Include toxic efficient model with inclusion of `RLHF`
- [ ] Include graph on trained weights to visualize certain stats on word-embeddings
- [ ] Include more base models as an adapter in this framework
