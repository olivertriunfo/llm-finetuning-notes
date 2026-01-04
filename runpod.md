# RunPod for LLM Training

RunPod provides cloud GPU infrastructure for training and deploying machine learning models, including Large Language Models.

## Why RunPod?

- **Cost-effective**: Lower prices than AWS/GCP for GPU instances
- **Dedicated GPUs**: No resource contention with other users
- **Easy setup**: Pre-configured templates available
- **Flexible**: Choose from various GPU types and configurations
- **Persistent storage**: Keep your data between sessions

## Getting Started

### Sign Up & Setup

1. Create an account at [https://www.runpod.io](https://www.runpod.io)
2. Add payment method
3. Create an API key from the Account Settings

### Creating a Pod

RunPod's basic unit is a "Pod" which is a container running on GPU hardware.

```bash
# Create a pod via API
curl https://api.runpod.io/v1/pods \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -X POST \
  -d '{
    "name": "llm-finetuning",
    "templateId": "nvidia-rtx4090",
    "gpuCount": 1,
    "minVcpu": 6,
    "minMemoryInGb": 22,
    "ports": ["8080"],
    "imageName": "pytorch/pytorch:latest",
    "env": {
      "HF_TOKEN": "your_huggingface_token"
    }
  }'
```

## Recommended GPU Configurations

| Use Case | GPU | vCPUs | RAM | Storage | Estimated Cost/hour |
|----------|-----|-------|-----|---------|---------------------|
| Small models (7B) | RTX 4090 | 6 | 22GB | 50GB | $0.30 - $0.50 |
| Medium models (13B) | RTX 4090 | 8 | 30GB | 100GB | $0.40 - $0.65 |
| Large models (30B+) | A100 40GB | 12 | 60GB | 200GB | $1.50 - $2.50 |
| Multi-GPU | A100 x4 | 48 | 240GB | 500GB | $5.00 - $8.00 |

### H200 and B200 GPUs

RunPod offers access to the latest NVIDIA GPUs including:

- **H200 (H100)**: Features 141GB VRAM with H100 architecture, ideal for large language models (70B+). The H200 pod type includes H100 GPUs with enhanced memory bandwidth and extreme throughput capabilities.

- **B200**: NVIDIA's next-generation GPU with Blackwell architecture, featuring 180GB VRAM. Offers maximum throughput for big models and significantly improved performance and memory efficiency.

**Specifications:**

| GPU | VRAM | System RAM | vCPUs | Description |
|-----|------|------------|-------|-------------|
| H200 | 141GB | 276GB | 24 | Extreme throughput for big models |
| B200 | 180GB | 283GB | 28 | Maximum throughput for big models |

Both GPU types are excellent choices for:
- Training very large language models (70B parameters and above)
- Research and development of next-generation AI models
- High-performance inference with low latency
- Multi-GPU distributed training

**Pricing Note:** Pricing for H200 and B200 GPUs varies based on availability and usage. Check the [RunPod Pricing Page](https://www.runpod.io/pricing) for current rates.

When creating a pod with these GPUs, use the appropriate template ID in the API request:

```bash
# Example: Create pod with H200 GPU
curl https://api.runpod.io/v1/pods \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -X POST \
  -d '{
    "name": "llm-h200-training",
    "templateId": "nvidia-h200",
    "gpuCount": 1,
    "minVcpu": 24,
    "minMemoryInGb": 276,
    "volumeSize": 300,
    "volumeMountPath": "/workspace",
    "imageName": "pytorch/pytorch:latest",
    "env": {
      "HF_TOKEN": "your_huggingface_token"
    }
  }'

# Example: Create pod with B200 GPU
curl https://api.runpod.io/v1/pods \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -X POST \
  -d '{
    "name": "llm-b200-training",
    "templateId": "nvidia-b200",
    "gpuCount": 1,
    "minVcpu": 28,
    "minMemoryInGb": 283,
    "volumeSize": 400,
    "volumeMountPath": "/workspace",
    "imageName": "pytorch/pytorch:latest"
  }'
```

## Using RunPod Templates

RunPod offers pre-configured templates for common ML frameworks:

- **Unsloth Template**: Ready-to-use environment with Unsloth installed
- **Hugging Face Template**: Includes transformers, datasets, and accelerate
- **LLM Training Template**: Optimized for LLM training with QLoRA

### Using the Unsloth Template

```bash
curl https://api.runpod.io/v1/pods \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -X POST \
  -d '{
    "name": "unsloth-finetuning",
    "templateId": "unsloth-llm-training",
    "gpuCount": 1,
    "minVcpu": 8,
    "minMemoryInGb": 30,
    "ports": ["8080"],
    "volumeSize": 100,
    "volumeMountPath": "/workspace"
  }'
```

## Training with Unsloth on RunPod

```python
# Connect to your RunPod via SSH or use the RunPod Web Terminal
# Clone your repository
!git clone https://github.com/your-repo/llm-finetuning
cd llm-finetuning

# Install requirements
!pip install unsloth peft transformers datasets torch

# Finetuning script
from unsloth import FastLanguageModel
from datasets import load_dataset
import torch

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-2-7b-hf",
    max_seq_length=2048,
)

# Load dataset
 dataset = load_dataset("my-dataset", split="train")

# Tokenize
train_dataset = tokenizer(dataset["text"], padding='max_length', truncation=True, return_tensors='pt')

# Setup QLoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

for epoch in range(3):
    outputs = model(**train_dataset)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Epoch {epoch}, Loss: {loss.item()}")

# Save to RunPod volume
model.save_pretrained("/workspace/llama-2-7b-finetuned")
tokenizer.save_pretrained("/workspace/llama-2-7b-finetuned")
```

## Cost Optimization Tips

### 1. Choose the Right GPU
- **RTX 4090**: Best value for 7-13B models
- **A100**: Needed for 30B+ models
- **L40S**: Good balance of cost and performance

### 2. Use Spot Instances

```json
{
  "name": "llm-spot",
  "templateId": "nvidia-rtx4090",
  "gpuCount": 1,
  "minVcpu": 6,
  "minMemoryInGb": 22,
  "spotPrice": 0.40,  # Max price per hour
  "maxSecondsRunning": 86400,  # 24 hours max
  "...": "..."
}
```

### 3. Save Checkpoints Frequently

```python
import os
from datetime import datetime

# Save checkpoint every epoch
checkpoint_path = f"/workspace/checkpoints/epoch_{epoch}"
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

model.save_pretrained(checkpoint_path)
tokenizer.save_pretrained(checkpoint_path)

# Save to Hugging Face Hub
model.push_to_hub("your-username/llama-2-7b-finetuned", use_auth_token=True)
```

### 4. Use Gradient Accumulation

```python
# Instead of larger batches, accumulate gradients
accumulation_steps = 4
optimizer.zero_grad()

for i in range(0, len(train_dataset), batch_size):
    batch = {k: v[i:i+batch_size] for k, v in train_dataset.items()}
    outputs = model(**batch)
    loss = outputs.loss / accumulation_steps
    loss.backward()

    if (i // batch_size) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## Monitoring Training

RunPod provides metrics and logging:

1. **RunPod Dashboard**: View GPU utilization, CPU, memory, and network
2. **Logs**: Access container logs in real-time
3. **Custom Metrics**: Send metrics from your training script

```python
import requests
import time

# Send custom metrics to RunPod
def send_metric(name, value):
    headers = {"Authorization": f"Bearer YOUR_API_KEY"}
    data = {
        "name": name,
        "value": value,
        "timestamp": int(time.time())
    }
    requests.post(
        f"https://api.runpod.io/v1/pods/{POD_ID}/metrics",
        headers=headers,
        json=data
    )

# Example usage in training loop
send_metric("training_loss", loss.item())
send_metric("learning_rate", optimizer.param_groups[0]['lr'])
```

## Deployment Options

After finetuning, deploy your model:

### Option 1: RunPod Endpoint

```bash
# Create an endpoint pod
curl https://api.runpod.io/v1/pods \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -X POST \
  -d '{
    "name": "llm-endpoint",
    "templateId": "nvidia-rtx4090",
    "gpuCount": 1,
    "minVcpu": 4,
    "minMemoryInGb": 16,
    "imageName": "runpod/llama-server",
    "env": {
      "MODEL_PATH": "/workspace/llama-2-7b-finetuned"
    },
    "ports": ["3000"]
  }'
```

### Option 2: Hugging Face Inference API

```python
from huggingface_hub import HfApi

api = HfApi()

# Create inference endpoint
api.create_repo(
    repo_id="your-username/llama-2-7b-finetuned",
    repo_type="model",
    exist_ok=True
)

# Upload your model
api.upload_folder(
    repo_id="your-username/llama-2-7b-finetuned",
    folder_path="/workspace/llama-2-7b-finetuned",
    repo_type="model"
)
```

## Best Practices

1. **Version Control**: Commit your training script and configuration
2. **Reproducibility**: Set random seeds
   ```python
   import torch
   import numpy as np

   torch.manual_seed(42)
   np.random.seed(42)
   ```
3. **Logging**: Use TensorBoard or Weights & Biases
4. **Backup**: Regularly save checkpoints to RunPod volume
5. **Monitor**: Watch GPU utilization to detect issues early
6. **Document**: Keep notes on hyperparameters and results

## Troubleshooting

### Pod Fails to Start
- Check GPU availability
- Verify your API key is valid
- Check your credit balance

### Training Hangs
- Check GPU memory usage in dashboard
- Reduce batch size
- Use gradient accumulation

### High Latency
- Use a RunPod template closer to your location
- Consider using RunPod's Edge Pods
- Check network configuration

## Resources

- [RunPod Documentation](https://docs.runpod.io/)
- [RunPod API Reference](https://api.runpod.io/docs)
- [RunPod Templates](https://www.runpod.io/templates)
- [RunPod Community](https://community.runpod.io/)
