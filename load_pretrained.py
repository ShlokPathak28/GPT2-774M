"""Load real OpenAI GPT-2 weights (via Hugging Face) into our GPTModel.

HF's GPT2Model stores QKV as one combined weight and uses Conv1D layers,
which are transposed relative to nn.Linear — so we split and transpose
as we copy each tensor across.

Usage:
    from transformers import GPT2Model
    from config import GPT_CONFIG_774M
    from model import GPTModel
    from load_pretrained import load_weights_into_gpt

    gpt2_hf = GPT2Model.from_pretrained("gpt2-large", cache_dir="checkpoints")
    model = GPTModel(GPT_CONFIG_774M)
    load_weights_into_gpt(model, gpt2_hf)

HF model name -> size:
    "gpt2"         -> 124M
    "gpt2-medium"  -> 355M
    "gpt2-large"   -> 774M
    "gpt2-xl"      -> 1558M
"""

import numpy as np
import torch


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: {left.shape} vs {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, gpt2_hf):
    d = gpt2_hf.state_dict()

    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, d["wpe.weight"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, d["wte.weight"])

    for b in range(len(gpt.trf_blocks)):
        prefix = f"h.{b}."

        q_w, k_w, v_w = np.split(d[prefix + "attn.c_attn.weight"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(d[prefix + "attn.c_attn.bias"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight, d[prefix + "attn.c_proj.weight"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias, d[prefix + "attn.c_proj.bias"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight, d[prefix + "mlp.c_fc.weight"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, d[prefix + "mlp.c_fc.bias"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight, d[prefix + "mlp.c_proj.weight"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias, d[prefix + "mlp.c_proj.bias"])

        gpt.trf_blocks[b].norm1.scale = assign(gpt.trf_blocks[b].norm1.scale, d[prefix + "ln_1.weight"])
        gpt.trf_blocks[b].norm1.shift = assign(gpt.trf_blocks[b].norm1.shift, d[prefix + "ln_1.bias"])
        gpt.trf_blocks[b].norm2.scale = assign(gpt.trf_blocks[b].norm2.scale, d[prefix + "ln_2.weight"])
        gpt.trf_blocks[b].norm2.shift = assign(gpt.trf_blocks[b].norm2.shift, d[prefix + "ln_2.bias"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, d["ln_f.weight"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, d["ln_f.bias"])
    gpt.out_head.weight = assign(gpt.out_head.weight, d["wte.weight"])  # weight tying