# GPT from scratch

A GPT-style transformer language model implemented from scratch in PyTorch —
architecture, training loop, sampling, checkpointing, and loading real
pretrained GPT-2 weights via Hugging Face.

Configurable across GPT-2's original sizes (124M / 355M / 774M / 1.5B) by
just swapping the config dict — the model code itself doesn't change.

## Structure

```
gpt-774m/
├── model.py            # GPTModel, TransformerBlock, MultiHeadAttention, LayerNorm, FeedForward
├── config.py            # GPT_CONFIG_124M / 355M / 774M / 1558M
├── dataset.py            # GPTDatasetV1, create_dataloader_v1
├── train.py              # calc_loss_batch, calc_loss_loader, train_model_simple
├── generate.py            # generate() — temperature + top-k sampling
├── load_pretrained.py      # Load real GPT-2 weights from Hugging Face into GPTModel
├── checkpoints/            # Saved .pth files (gitignored)
├── data/                  # Training text, e.g. the-verdict.txt (gitignored)
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Requires a CUDA-enabled PyTorch build to train/run on GPU — see
https://pytorch.org/get-started/locally/ for the right install command for
your CUDA version.

## Usage

Train from scratch on your own text:

```python
from config import GPT_CONFIG_355M
from model import GPTModel
from dataset import create_dataloader_v1
from train import train_model_simple
import torch

model = GPTModel(GPT_CONFIG_355M)
# ... build dataloaders, optimizer, then call train_model_simple(...)
```

Load real pretrained GPT-2 weights instead:

```python
from load_pretrained import load_weights_into_gpt
from transformers import GPT2Model

gpt2_hf = GPT2Model.from_pretrained("gpt2-large", cache_dir="checkpoints")
model = GPTModel(GPT_CONFIG_774M)
load_weights_into_gpt(model, gpt2_hf)
```

Generate text:

```python
from generate import generate
# generate(model, idx, max_new_tokens, context_size, temperature=0.7, top_k=25)
```

## Notes

- Trained from scratch on a small dataset, the model will overfit fast —
  this is expected, not a bug. Loading real GPT-2 weights is the path to
  actually coherent generation without needing a huge training corpus.
- 8GB VRAM comfortably handles inference up to GPT-2-large (774M) and
  training up to roughly GPT-2-medium (355M) at small batch sizes.