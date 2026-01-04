# Quick Start Guide (2026)

This guide provides a fast path to finetuning LLMs using the tools documented in this repository.

## 1. Setup Your Environment

### On RunPod

```bash
# Create a pod with 1-2 RTX 4090 GPUs
curl https://api.runpod.io/v1/pods \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -X POST \
  -d '{
    "name": "llm-finetuning-2026",
    "templateId": "unsloth-llm-training",
    "gpuCount": 1,
    "minVcpu": 8,
    "minMemoryInGb": 30,
    "volumeSize": 100,
    "volumeMountPath": "/workspace"
  }'
```

### Local Setup

```bash
pip install unsloth peft transformers datasets torch
```

## 2. Finetune with Unsloth + LoRA/QLoRA

Unsloth supports both **Standard LoRA** and **QLoRA** (Quantized LoRA). Choose based on your needs:

### Option A: QLoRA (Recommended for Large Models)

```python
from unsloth import FastLanguageModel
from datasets import load_dataset
import torch

# Load model WITH 4-bit quantization (QLoRA)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3.1-70B-hf",
    max_seq_length=8192,  # 2026: longer contexts
    load_in_4bit=True,    # Enable 4-bit quantization
    quantization_type="bnb",
)

# Load dataset
dataset = load_dataset("my-dataset", split="train")

# Tokenize with instruction format
def tokenize_function(examples):
    return tokenizer(
        f"### Instruction: {examples['instruction']}\n\n### Input: {examples['input']}\n\n### Response: {examples['output']}",
        truncation=True,
        padding='max_length',
        max_length=4096
    )

train_dataset = dataset.map(tokenize_function, batched=True)

# Apply QLoRA (r=64 for 70B model)
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# Train
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

for epoch in range(3):
    model.train()
    for step, batch in enumerate(train_dataset):
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 10 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")

# Save QLoRA adaptation
model.save_pretrained("/workspace/llama-3.1-70b-qlora-2026")
```

### Option B: Standard LoRA (For Maximum Accuracy)

```python
from unsloth import FastLanguageModel
from datasets import load_dataset
import torch

# Load model WITHOUT quantization (Standard LoRA)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3.1-70B-hf",
    max_seq_length=8192,
    load_in_4bit=False,  # Disable quantization for standard LoRA
)

# Load dataset
dataset = load_dataset("my-dataset", split="train")

# Tokenize
train_dataset = dataset.map(tokenize_function, batched=True)

# Apply Standard LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# Train
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

for epoch in range(3):
    model.train()
    for step, batch in enumerate(train_dataset):
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 10 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")

# Save LoRA adaptation
model.save_pretrained("/workspace/llama-3.1-70b-lora-2026")
```

## 3. Deploy Your Model

### Option A: RunPod Endpoint

```bash
curl https://api.runpod.io/v1/pods \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -X POST \
  -d '{
    "name": "llm-endpoint-2026",
    "templateId": "nvidia-rtx4090",
    "gpuCount": 1,
    "minVcpu": 4,
    "minMemoryInGb": 16,
    "imageName": "runpod/llama-server",
    "env": {
      "MODEL_PATH": "/workspace/llama-3.1-70b-lora-2026"
    },
    "ports": ["3000"]
  }'
```

### Option B: Hugging Face Hub

```python
from huggingface_hub import HfApi

api = HfApi()
api.create_repo(
    repo_id="your-username/llama-3.1-70b-lora-2026",
    repo_type="model",
    exist_ok=True
)

model.push_to_hub("your-username/llama-3.1-70b-lora-2026")
tokenizer.push_to_hub("your-username/llama-3.1-70b-lora-2026")
```

## 4. Inference

```python
from peft import PeftModel
from unsloth import FastLanguageModel
import torch

# Load base model
base_model, tokenizer = FastLanguageModel.from_pretrained(
    "meta-llama/Llama-3.1-70B-hf"
)

# Load LoRA/QLoRA
model = PeftModel.from_pretrained(base_model, "/workspace/llama-3.1-70b-lora-2026")

# Generate
prompt = "Explain quantum computing in simple terms:"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Recommended Workflow

1. **Start small**: Test with a 7B model before scaling to 70B
2. **Choose your approach**: Use QLoRA for fast training, Standard LoRA for maximum accuracy
3. **Use existing datasets**: Try Alpaca, Guanaco, or ShareGPT first
4. **Monitor training**: Use WandB or TensorBoard
5. **Save checkpoints**: Every 500 steps
6. **Evaluate**: On a held-out validation set
7. **Iterate**: Adjust LoRA rank, learning rate, and batch size

## Cost Estimation (RunPod, 2026)

| Configuration | Estimated Cost/Hour | Training Time (3 epochs) |
|---------------|---------------------|--------------------------|
| RTX 4090, 7B model | $0.35 | 6-12 hours |
| RTX 4090, 13B model | $0.45 | 12-24 hours |
| RTX 4090, 30B model | $0.55 | 24-48 hours |
| A100 40GB, 70B model | $1.80 | 24-48 hours |

## Next Steps

- 📖 Read [unsloth.md](unsloth.md) for detailed Unsloth usage (QLoRA vs LoRA comparison)
- 📊 Check [datasets.md](datasets.md) for dataset selection
- 🏃 Learn [loras.md](loras.md) for advanced LoRA techniques
- ☁️ Explore [runpod.md](runpod.md) for deployment options

## Common Commands

```bash
# Clone this repo
git clone https://github.com/your-repo/llm-finetuning-notes.git

# Install dependencies
pip install -r requirements.txt

# Connect to RunPod
ssh -p 32111 root@<RUNPOD_IP>

# Monitor GPU usage
nvidia-smi

# Check disk space
df -h

# Save to Hugging Face Hub
huggingface-cli login
model.push_to_hub("your-model")
```

## Troubleshooting

### Training too slow?
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision (fp16/bf16)
- Try QLoRA instead of Standard LoRA

### Out of Memory?
- Reduce batch size to 1
- Increase gradient accumulation steps
- Use QLoRA (requires less memory)
- Use a smaller model

### Poor results?
- Increase LoRA rank (try r=64)
- Train longer (more epochs)
- Try a different learning rate (2e-4 to 5e-4)
- Check your dataset quality

### Need help?
- Check the RunPod logs
- Review the training metrics
- Consult the [LoRA](loras.md) guide for tuning tips
