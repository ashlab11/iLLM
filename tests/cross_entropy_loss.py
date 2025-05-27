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

#Link to discussion on ChatGPT:
#https://chatgpt.com/share/68351df6-5774-800c-a5ef-0ee70e74d9cc