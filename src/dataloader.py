from torch.utils.data import DataLoader


class EnDeDataLoader:
    def __init__(self, dataset, tokenizer, batch_size=32, shuffle=True):
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.shuffle = shuffle

    def collate_fn(self, batch):
        src_texts = [item['translation']['de'] for item in batch]
        tgt_texts = [item['translation']['en'] for item in batch]

        src_encodings = self.tokenizer(src_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        tgt_encodings = self.tokenizer(tgt_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

        return {
            'src_input_ids': src_encodings['input_ids'],
            'src_attention_mask': src_encodings['attention_mask'],
            'tgt_input_ids': tgt_encodings['input_ids'][:,:-1], # everything except last
            'tgt_labels': tgt_encodings['input_ids'][:,1:],  # Shift labels for teacher forcing
            'tgt_attention_mask': tgt_encodings['attention_mask'][:,:-1],  # Shift attention mask for teacher forcing
        }

    def get_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.collate_fn)