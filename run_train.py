import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import urllib.request
import torch

from config import GPT_CONFIG_774M
from model import GPTModel
from dataset import create_dataloader_v1
from train import train_model_simple
import bitsandbytes as bnb
import tiktoken

# 1. Download "The Verdict" if not already present
file_path = "data/the-verdict.txt"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

if not os.path.exists(file_path):
    os.makedirs("data", exist_ok=True)
    urllib.request.urlretrieve(url, file_path)

with open(file_path, "r", encoding="utf-8") as f:
    text_data = f.read()

print("Total characters:", len(text_data))

# 2. Train/val split
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

torch.manual_seed(123)

train_loader = create_dataloader_v1(
    train_data,
    batch_size=1,               # small batch — 774M is memory-hungry
    max_length=GPT_CONFIG_774M["context_length"] // 8,   # 128 — keep short for VRAM
    stride=GPT_CONFIG_774M["context_length"] // 8,
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=1,
    max_length=GPT_CONFIG_774M["context_length"] // 8,
    stride=GPT_CONFIG_774M["context_length"] // 8,
    drop_last=False,
    shuffle=False,
    num_workers=0
)

print("Train batches:", len(train_loader))
print("Val batches:", len(val_loader))

# 3. Build model, move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = GPTModel(GPT_CONFIG_774M)
model.to(device)

print(f"Allocated after model load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
tokenizer = tiktoken.get_encoding("gpt2")

# 4. Train
num_epochs = 3   # keep short for the VRAM/sanity test
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=1,
    start_context="Every effort moves you", tokenizer=tokenizer
)

print(f"Peak allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")