from torch.utils.data import Dataset

class WikiDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        with open(self.file_path, 'r', encoding='utf-8') as file:
            for line in file:
                sentence = line.strip()
                self.data.append(sentence)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence = self.data[index]
        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        return input_ids, attention_mask