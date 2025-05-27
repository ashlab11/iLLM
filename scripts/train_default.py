import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import sentencepiece as spm
from datasets import load_dataset
from dataset import StreamTextDataset
from src.models.testllm import TestLLM


ds = StreamTextDataset(
    file_path='data/slimpajama_sample.txt',
    tokenizer_model='tokenizer/tokenizer.model' 
)
loader = DataLoader(ds, batch_size = 1, num_workers = 0, shuffle = False)

tokenizer = spm.SentencePieceProcessor(model_file='tokenizer/tokenizer.model')
vocab_size = tokenizer.vocab_size()
model = TestLLM(
    vocab_size = vocab_size,
    embed_size = 512, 
    num_mha = 4, 
    num_heads = 16, 
    ff_size = 2048
)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
device = torch.device('mps' if torch.mps.is_available() else 'cpu')
model.to(device)
lens = []

for idx, batch in enumerate(loader):
    if idx >= 100000:
        break
    if len(batch[0]) == 1:
        print(batch)
    lens.append(len(batch[0]))
    """batch = batch.to(device)
    logits = model(batch)
    loss = criterion(logits)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if idx % 1000 == 0:
        print(f"Batch {idx}, Loss: {loss.item()}")"""

print(min(lens))