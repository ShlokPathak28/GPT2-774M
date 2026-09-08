import torch
from transformers import GPT2Model

from config import GPT_CONFIG_774M
from model import GPTModel
from load_pretrained import load_weights_into_gpt
from generate import generate
import tiktoken

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Download real GPT-2-large weights (this is a few GB, first run will take a bit)
print("Downloading GPT-2-large weights...")
gpt2_hf = GPT2Model.from_pretrained("gpt2-large", cache_dir="checkpoints")
gpt2_hf.eval()

# Build our model and copy the weights in
model = GPTModel(GPT_CONFIG_774M)
load_weights_into_gpt(model, gpt2_hf)
model.to(device)
model.eval()

print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB" if device.type == "cuda" else "Running on CPU")

# Generate
tokenizer = tiktoken.get_encoding("gpt2")

torch.manual_seed(123)
token_ids = generate(
    model=model,
    idx=torch.tensor(tokenizer.encode("Every effort moves you")).unsqueeze(0).to(device),
    max_new_tokens=40,
    context_size=GPT_CONFIG_774M["context_length"],
    top_k=50,
    temperature=1.0
)

print(tokenizer.decode(token_ids.squeeze(0).tolist()))

print("Type a prompt (or 'quit' to exit):\n")

while True:
    prompt = input("You: ")
    if prompt.lower() == "quit":
        break

    torch.manual_seed(123)
    token_ids = generate(
        model=model,
        idx=torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device),
        max_new_tokens=60,
        context_size=GPT_CONFIG_774M["context_length"],
        top_k=50,
        temperature=0.8
    )
    output = tokenizer.decode(token_ids.squeeze(0).tolist())
    print("\nModel:", output, "\n")