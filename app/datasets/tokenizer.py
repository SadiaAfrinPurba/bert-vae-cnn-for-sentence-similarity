from transformers import BertTokenizer


def get_bert_tokenizer(model_name: str = 'bert-base-uncased'):
    tokenizer = BertTokenizer.from_pretrained(model_name)

    return tokenizer
