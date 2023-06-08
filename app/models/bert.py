from transformers import BertModel


def get_bert_model(model_name: str = "bert-base-uncased"):
    model = BertModel.from_pretrained(model_name)

    return model
