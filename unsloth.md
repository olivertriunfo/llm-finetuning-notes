# Unsloth: Fast LLM Finetuning

Unsloth is a library that enables extremely fast finetuning of LLMs using **QLoRA (Quantization-aware Low-Rank Adaptation)** and standard **LoRA (Low-Rank Adaptation)**.

## Features

- **10x faster** than traditional finetuning
- **4-bit quantization** with minimal accuracy loss (QLoRA)
- **Standard LoRA** support for non-quantized models
- **Memory-efficient** training
- **GPU-friendly** - works with consumer GPUs
- **Easy integration** with Hugging Face Transformers

## Installation

```bash
pip install unsloth
```

## Quick Example

```python
from unsloth import FastLanguageModel
import torch

# Load model with 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-2-7b-hf",
    max_seq_length=2048,
)

# Prepare dataset
dataset = ["Hello, how are you?", "What is AI?", "Explain machine learning."]

# Tokenize
train_dataset = tokenizer(dataset, padding='max_length', truncation=True, return_tensors='pt')

# Convert to QLoRA format
from unsloth import FastLanguageModel
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# Training setup
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

# Training loop
for epoch in range(3):
    outputs = model(**train_dataset)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Epoch {epoch}, Loss: {loss.item()}")

# Save the adapted model
model.save_pretrained("llama-2-7b-finetuned")
tokenizer.save_pretrained("llama-2-7b-finetuned")
```

## Key Parameters

### Model Loading
- `model_name`: Hugging Face model name
- `max_seq_length`: Maximum sequence length (default: 2048)
- `dtype`: Data type (default: `None` for auto detection)
- `load_in_4bit`: Boolean (default: `True` for QLoRA, `False` for standard LoRA)

### LoRA Configuration (QLoRA and Standard LoRA)
- `r`: Rank of LoRA matrices (default: 16)
- `target_modules`: List of modules to apply LoRA (e.g., `["q_proj", "k_proj", "v_proj", "o_proj"]`)
- `lora_alpha`: Alpha parameter for LoRA scaling
- `lora_dropout`: Dropout rate for LoRA layers
- `bias`: Bias type ("none", "all", or "lora_only")

### QLoRA-Specific Parameters
- `quantization_type`: "bnb" for 4-bit quantization
- `compute_dtype`: Computation dtype (e.g., `torch.float16` or `torch.bfloat16`)

## Training Tips

1. **Batch Size**: Start with batch size 1-4, increase gradually
2. **Learning Rate**: Typical range is 1e-4 to 5e-4
3. **Epochs**: 3-5 epochs often sufficient for finetuning
4. **Gradient Accumulation**: Use if you get OOM errors
5. **Mixed Precision**: Use `torch.float16` or `torch.bfloat16` when possible

## Common Issues

### Out of Memory (OOM)
```python
# Reduce batch size
batch_size = 1

# Use gradient accumulation
accumulation_steps = 4
```

### Slow Training
```python
# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use mixed precision
model = model.to(torch.float16)
```

### Low Accuracy
```python
# Increase LoRA rank
r = 32  # from 16

# Train longer
num_epochs = 5  # from 3
```

## Advanced Usage

### Standard LoRA (Non-Quantized)

```python
from unsloth import FastLanguageModel
import torch

# Load model WITHOUT quantization for standard LoRA
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-2-7b-hf",
    max_seq_length=2048,
    load_in_4bit=False,  # Disable quantization for standard LoRA
)

# Prepare dataset
dataset = ["Hello, how are you?", "What is AI?", "Explain machine learning."]
train_dataset = tokenizer(dataset, padding='max_length', truncation=True, return_tensors='pt')

# Apply standard LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
)

# Training
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

for epoch in range(3):
    outputs = model(**train_dataset)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Epoch {epoch}, Loss: {loss.item()}")

# Save LoRA adaptation
model.save_pretrained("llama-2-7b-lora-standard")
tokenizer.save_pretrained("llama-2-7b-lora-standard")
```

### QLoRA (Quantized)

```python
from unsloth import FastLanguageModel
import torch

# Load model WITH 4-bit quantization for QLoRA
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-2-7b-hf",
    max_seq_length=2048,
    load_in_4bit=True,  # Enable quantization for QLoRA
    quantization_type="bnb",
)

# Prepare dataset
dataset = ["Hello, how are you?", "What is AI?", "Explain machine learning."]
train_dataset = tokenizer(dataset, padding='max_length', truncation=True, return_tensors='pt')

# Apply QLoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
)

# Training
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

for epoch in range(3):
    outputs = model(**train_dataset)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Epoch {epoch}, Loss: {loss.item()}")

# Save QLoRA adaptation
model.save_pretrained("llama-2-7b-lora-q4bit")
tokenizer.save_pretrained("llama-2-7b-lora-q4bit")
```

### Custom Dataset

```python
from datasets import load_dataset

# Load a Hugging Face dataset
dataset = load_dataset("imdb", split="train")

# Tokenize
def tokenize_function(examples):
    return tokenizer(examples["text"], padding='max_length', truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Convert to torch dataset
train_dataset = tokenized_dataset.to_torch()
```

### Evaluation

```python
from unsloth import FastLanguageModel

model.eval()
with torch.no_grad():
    outputs = model.generate(
        input_ids=tokenizer("Hello, how are you?", return_tensors="pt").input_ids,
        max_length=50,
        temperature=0.7,
        top_p=0.9,
    )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## LoRA vs QLoRA Comparison

| Feature | Standard LoRA | QLoRA |
|---------|---------------|-------|
| Model Size | Full 16-bit model | 4-bit quantized model |
| Memory Usage | High | Very Low |
| Training Speed | Fast | Very Fast (10x) |
| GPU Required | 24GB+ | 12GB+ |
| Accuracy | High | Near-High |
| Use Case | When you need maximum accuracy | Fast experimentation, large models |
| File Size | ~40GB (7B) | ~4GB (7B) |

## Resources

- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
