from torch.utils.data import Dataset

class WikiDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length, is_taking_subset, data_subset_size):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_taking_subset = is_taking_subset
        self.data_subset_size = data_subset_size
        self.data = []

        with open(self.file_path, 'r', encoding='utf-8') as file:
            for line in file:
                sentence = line.strip()
                self.data.append(sentence)
        if self.is_taking_subset:
            self.data = self.data[:data_subset_size]

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
        