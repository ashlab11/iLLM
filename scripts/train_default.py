import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import sentencepiece as spm
from datasets import load_dataset
from tokenizer.dataset import StreamTextDataset
from src.models.testllm import TestLLM
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

ds = StreamTextDataset(
    file_path='data/slimpajama_sample_sorted.txt',
    tokenizer_model='tokenizer/tokenizer.model' 
)

def collate_fn(batch):
    """Collate function to pad sequences in a batch."""
    return pad_sequence(batch, batch_first=True, padding_value=0)

batch_size = 2
loader = DataLoader(ds, batch_size = batch_size, num_workers = 0, shuffle = False, collate_fn=collate_fn)

tokenizer = spm.SentencePieceProcessor(model_file='tokenizer/tokenizer.model')
vocab_size = tokenizer.vocab_size()
params = {
    'vocab_size': vocab_size,
    'embed_size': 512,
    'num_mha': 96,
    'num_heads': 64,
    'ff_size': 1024
}

model = TestLLM(**params)

#Print total model parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")


optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
device = torch.device('mps' if torch.mps.is_available() else 'cpu')
print(f"Using device: {device}")
model.to(device)

#Getting id for EOS token
eos_id = tokenizer.piece_to_id('<eos>')
eos_col = torch.full((batch_size, 1), eos_id, dtype=torch.long, device=device)
losses = []

for idx, batch in tqdm(enumerate(loader), total = 1440482 // batch_size):
    batch = batch.to(device)
    #Shifting batch for next token prediction
    targets = torch.cat((batch[:, 1:], eos_col), dim = 1)
    
    mask = batch == 0
    logits = model(batch, mask).transpose(1, 2)  # Transpose to match (batch_size, vocab_size, seq_length)
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if idx % 100 == 0:
        losses.append(loss.item())
        print(f"Batch {idx}, Loss: {loss.item()}")
    if idx % 10000 == 0:
        print(f"Saving model at batch {idx}")
        # Save the model periodically
        torch.save(model.state_dict(), f'src/models/testllm.pth')