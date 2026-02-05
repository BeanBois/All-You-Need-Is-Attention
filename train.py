from datasets import load_dataset
from transformers import AutoTokenizer
from src.dataloader import EnDeDataLoader
from src.transformer import Transformer

import torch 
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

import matplotlib.pyplot as plt
tokenizer = AutoTokenizer.from_pretrained("t5-base")  # or any model with EN-DE support
dataset = load_dataset("wmt14", "de-en")


train_dataset = dataset['train']  # Using a subset for quick testing
val_dataset = dataset['validation']

train_loader = EnDeDataLoader(train_dataset, tokenizer, batch_size=16).get_dataloader()
val_loader = EnDeDataLoader(val_dataset, tokenizer, batch_size=16, shuffle=False).get_dataloader()

model = Transformer(src_vocab_size=tokenizer.vocab_size, tgt_vocab_size=tokenizer.vocab_size)
loss_fn = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = AdamW(model.parameters(), lr=1e-4)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            src = batch['src_input_ids'].to(device)
            tgt = batch['tgt_input_ids'].to(device)
            labels = batch['tgt_labels'].to(device)
            
            outputs = model(src, tgt)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), labels.reshape(-1))
            total_loss += loss.item()
    
    model.train()
    return total_loss / len(val_loader)

if __name__ == "__main__":
    num_epochs = 20
    model.train()
    training_losses = []
    validation_losses = []
    for epoch in range(num_epochs):
        total_training_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            src_input_ids = batch['src_input_ids'].to(device)
            tgt_input_ids = batch['tgt_input_ids'].to(device)
            labels = batch['tgt_labels'].to(device)


            outputs = model(src_input_ids, tgt_input_ids, train=True)
            loss = loss_fn(outputs.contiguous().view(-1, outputs.size(-1)), labels.contiguous().view(-1))
            loss.backward()
            optimizer.step()

            total_training_loss += loss.item()
        avg_training_loss = total_training_loss / len(train_loader)
        training_losses.append(avg_training_loss)
        val_loss = validate(model, val_loader, loss_fn, device)
        validation_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss}")
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_training_loss}")
    torch.save(model.state_dict(), "transformer_model.pth")

    # Plot training and validation loss
    plt.plot(range(1, num_epochs + 1), training_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), validation_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss over Epochs')
    plt.savefig('loss_plot.png')    