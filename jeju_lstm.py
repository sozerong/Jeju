import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import pandas as pd
import numpy as np
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Set up environment variables for distributed training
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'  # Adjust according to the available GPUs
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

# Define the model and dataset classes (assuming tokenize function remains unchanged)
class Seq2SeqDataset(Dataset):
    def __init__(self, encoder_input, decoder_input, decoder_target):
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.decoder_target = decoder_target

    def __len__(self):
        return len(self.encoder_input)

    def __getitem__(self, idx):
        return (self.encoder_input[idx], self.decoder_input[idx], self.decoder_target[idx])

class Seq2SeqModel(nn.Module):
    def __init__(self, std_vocab_size, jej_vocab_size, hidden_size):
        super(Seq2SeqModel, self).__init__()
        self.encoder = nn.LSTM(input_size=std_vocab_size, hidden_size=hidden_size, batch_first=True)
        self.decoder = nn.LSTM(input_size=jej_vocab_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, jej_vocab_size)
    
    def forward(self, encoder_input, decoder_input):
        _, (hidden, cell) = self.encoder(encoder_input)
        decoder_output, _ = self.decoder(decoder_input, (hidden, cell))
        output = self.fc(decoder_output)
        return output

# Initialize the process group for distributed training
dist.init_process_group(backend='nccl', init_method='env://')
train_df = 'C:/webframe/Git/Jeju/train_df.csv'
# Load and preprocess the data
df = train_df[:4000]
encoder_input, decoder_input, decoder_target, std_vocab_size, jej_vocab_size, std_to_index, jej_to_index = tokenize(df)

# Convert the data to PyTorch tensors
encoder_input = torch.tensor(encoder_input, dtype=torch.float32)
decoder_input = torch.tensor(decoder_input, dtype=torch.float32)
decoder_target = torch.tensor(decoder_target, dtype=torch.long)

# Create DataLoader with DistributedSampler
dataset = Seq2SeqDataset(encoder_input, decoder_input, decoder_target)
sampler = DistributedSampler(dataset)
train_loader = DataLoader(dataset, batch_size=64, shuffle=False, sampler=sampler)

# Define the model, loss function, and optimizer
hidden_size = 1024
model = Seq2SeqModel(std_vocab_size, jej_vocab_size, hidden_size)

# Move model to the correct device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Wrap model with DistributedDataParallel
model = DDP(model)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(train_loader, model, criterion, optimizer, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        for encoder_input, decoder_input, decoder_target in train_loader:
            encoder_input = encoder_input.to(device)
            decoder_input = decoder_input.to(device)
            decoder_target = decoder_target.to(device)

            optimizer.zero_grad()
            output = model(encoder_input, decoder_input)
            loss = criterion(output.permute(0, 2, 1), decoder_target)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    return model

# Train the model
model = train_model(train_loader, model, criterion, optimizer, num_epochs=100)

# Save the model
torch.save(model.state_dict(), 'std_to_jej.pth')

# Define the inference function
def decode_seq(model, input_seq):
    model.eval()
    with torch.no_grad():
        input_seq = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
        _, (hidden, cell) = model.module.encoder(input_seq)
        target_seq = torch.zeros((1, 1, jej_vocab_size), dtype=torch.float32).to(device)
        target_seq[0, 0, jej_to_index['<start>']] = 1

        decoded_sent = ""
        stop = False
        while not stop:
            output, (hidden, cell) = model.module.decoder(target_seq, (hidden, cell))
            output = model.module.fc(output)
            token_index = output.argmax(2).item()
            pred_char = index_to_jej[token_index]
            decoded_sent += pred_char
            
            if (pred_char == "<end>" or len(decoded_sent) > 373):
                stop = True
                
            target_seq = torch.zeros((1, 1, jej_vocab_size), dtype=torch.float32).to(device)
            target_seq[0, 0, token_index] = 1
    
    return decoded_sent

# Evaluate the model on a few sequences
for seq_index in [1, 50, 100, 200, 300]:
    input_seq = encoder_input[seq_index].unsqueeze(0).numpy()
    decoded_seq = decode_seq(model, input_seq)
    
    print("입력문장:", train_df['표준어'][seq_index])
    print("정답:", train_df['제주어'][seq_index][1:len(train_df['제주어'][seq_index])-1])  # "<start>", "<end>" 제거
    print("번역기:", decoded_seq[:len(decoded_seq)-1])
    print("\n")

# Clean up
dist.destroy_process_group()
