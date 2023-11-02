from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn as nn


def load_model(model_name: str = 'bert-base-uncased'):

    # load the model
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.classifier = nn.Linear(768, 3)

    # load the corresponding tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer
