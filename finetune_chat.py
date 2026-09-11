"""
LoRA instruction-tune GPT-2-large so it behaves more like a chat assistant
instead of a plain text-completion model.

Uses Hugging Face's off-the-shelf gpt2-large (same weights you already
loaded into your custom GPTModel via load_pretrained.py) so it plugs
directly into peft/trl without needing to adapt your from-scratch class.

Run locally (Python 3.12 venv) or on Colab.

Setup (PINNED versions, do not use -U/latest):
    pip install transformers==4.46.1 peft==0.13.2 trl==0.12.0 accelerate==0.34.2 datasets==3.0.1

Do NOT install bitsandbytes — not needed here, and its triton dependency is
broken in current environments.
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "gpt2-large"
OUTPUT_DIR = "gpt2-large-chat-lora"

# ---------------------------------------------------------------------------
# 1. Load a small instruction dataset (Dolly-15k) and format it
# ---------------------------------------------------------------------------
raw_dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

# Keep it manageable for a quick local/Colab run — subsample to 3000 examples
raw_dataset = raw_dataset.shuffle(seed=42).select(range(3000))

PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Response:
{response}"""


def format_example(example):
    instruction = example["instruction"]
    if example.get("context"):
        instruction = f"{instruction}\n\nContext: {example['context']}"
    text = PROMPT_TEMPLATE.format(instruction=instruction, response=example["response"])
    return {"text": text}


dataset = raw_dataset.map(format_example, remove_columns=raw_dataset.column_names)
dataset = dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

print(f"Train examples: {len(train_dataset)}")
print(f"Eval examples: {len(eval_dataset)}")

# ---------------------------------------------------------------------------
# 2. Load base model + tokenizer
# ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token by default

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# ---------------------------------------------------------------------------
# 3. Apply LoRA — GPT-2 uses Conv1D layers (not nn.Linear), target by name
# ---------------------------------------------------------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["c_attn", "c_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # should show a tiny fraction of 774M

# ---------------------------------------------------------------------------
# 4. Train
# ---------------------------------------------------------------------------
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    dataset_text_field="text",
    max_seq_length=512,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    bf16=True,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()

# ---------------------------------------------------------------------------
# 5. Save the LoRA adapter
# ---------------------------------------------------------------------------
model.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
print(f"Saved LoRA adapter to {OUTPUT_DIR}/final_adapter")

# ---------------------------------------------------------------------------
# 6. Quick test
# ---------------------------------------------------------------------------
test_prompts = [
    "What is the capital of France?",
    "Write a short poem about the ocean.",
    "Explain how photosynthesis works in simple terms.",
]

model.eval()
for instr in test_prompts:
    prompt = f"### Instruction:\n{instr}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=100, temperature=0.7, top_k=50, do_sample=True
        )
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\n### Instruction:\n{instr}\n### Response:\n{response}")
