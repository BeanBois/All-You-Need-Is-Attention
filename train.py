from datasets import load_dataset
from transformers import AutoTokenizer
from src.dataloader import EnDeDataLoader
from src.transformer import Transformer

import torch 
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

tokenizer = AutoTokenizer.from_pretrained("t5-base")  # or any model with EN-DE support
dataset = load_dataset("wmt14", "de-en")


train_dataset = dataset['train'].select(range(1000))  # Using a subset for quick testing
val_dataset = dataset['validation'].select(range(200))

train_loader = EnDeDataLoader(train_dataset, tokenizer, batch_size=16).get_dataloader()
val_loader = EnDeDataLoader(val_dataset, tokenizer, batch_size=16, shuffle=False).get_dataloader()

model = Transformer(src_vocab_size=tokenizer.vocab_size, tgt_vocab_size=tokenizer.vocab_size)
losses = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = AdamW(model.parameters(), lr=1e-4)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


if __name__ == "__main__":
    num_epochs = 3
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            src_input_ids = batch['src_input_ids'].to(device)
            tgt_input_ids = batch['tgt_input_ids'].to(device)
            labels = batch['tgt_labels'].to(device)


            outputs = model(src_input_ids, tgt_input_ids, train=True)
            loss = losses(outputs.contiguous().view(-1, outputs.size(-1)), labels.contiguous().view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss}")