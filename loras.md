# LoRAs: Low-Rank Adaptation for Efficient LLM Finetuning

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that reduces memory and computation costs while maintaining model performance. This is especially important for 2026 LLMs with billions of parameters.

## What is LoRA?

LoRA works by:
1. Freezing the pre-trained model weights
2. Injecting trainable low-rank matrices into each layer
3. Only training these low-rank matrices (typically <1% of total parameters)
4. Merging the adaptations back into the model for inference

### Key Advantages

- **Memory Efficient**: 10-100x fewer trainable parameters
- **Faster Training**: Reduced computation
- **Easy Transfer**: Share LoRA adaptations without full model
- **Compatibility**: Works with 4-bit quantization (QLoRA)
- **Stackable**: Multiple LoRAs can be combined

## Mathematical Formulation

For a weight matrix W₀ ∈ ℝᵈᵢ×ᵈₒ (original weights), LoRA replaces it with:

W₀ + ΔW = W₀ + BA

Where:
- B ∈ ℝᵈᵢ×ᵣ
- A ∈ ℝᵣ×ᵈₒ
- r << min(dᵢ, dₒ) (rank, typically 4-256)

Only B and A are trained, not W₀.

## Using LoRA with Unsloth (2026 Edition)

### Basic LoRA Finetuning

```python
from unsloth import FastLanguageModel
from datasets import load_dataset
import torch

# Load model with 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3.1-70B-hf",
    max_seq_length=8192,  # 2026: longer contexts
)

# Load dataset
dataset = load_dataset("my-dataset", split="train")

# Tokenize
train_dataset = tokenizer(dataset["text"], padding='max_length', truncation=True, return_tensors='pt')

# Setup LoRA
model = FastLanguageModel.get_peft_model(
    model,  # your model
    r=64,    # LoRA rank (2026: higher ranks for larger models)
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha=128,      # Alpha scaling
    lora_dropout=0.1,     # Dropout
    bias="none",         # No bias
    use_gradient_checkpointing="unsloth",  # Advanced checkpointing
)

# Training setup
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

# Training loop
for epoch in range(5):  # 2026: more epochs with LoRA
    outputs = model(**train_dataset)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Save LoRA adaptation
model.save_pretrained("llama-3.1-70b-lora")
```

## LoRA Configuration Guide

### Choosing the Right Rank (r)

| Model Size | Recommended Rank | Use Case |
|------------|------------------|----------|
| 7B | 8-32 | General finetuning, quick experiments |
| 13B | 16-64 | Domain adaptation, better performance |
| 30B-70B | 32-128 | High-quality finetuning |
| 70B+ | 64-256 | Specialized applications |

```python
# Example: High-rank LoRA for large model
model = FastLanguageModel.get_peft_model(
    model,
    r=128,  # Higher rank for better capacity
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=256,
    lora_dropout=0.05,
)
```

### Target Modules

Common modules to apply LoRA:

- **Attention**: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- **Feed Forward**: `gate_proj`, `up_proj`, `down_proj`
- **LM Head**: `embed_tokens` (for generation tasks)

```python
# Full module coverage
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "embed_tokens"
]
```

### Alpha Scaling

Alpha (lora_alpha) scales the LoRA update:

```python
lora_alpha = r * scaling_factor
```

Common ratios:
- `lora_alpha = 2 * r` (moderate scaling)
- `lora_alpha = r` (default)
- `lora_alpha = 4 * r` (aggressive scaling for hard tasks)

## Advanced LoRA Techniques

### Multi-Step LoRA

Train multiple LoRAs sequentially and merge them:

```python
# Step 1: Basic finetuning
model.step1 = FastLanguageModel.get_peft_model(model, r=32, ...)
model.step1.train(...)  # Train on general dataset

# Step 2: Domain specialization
model.step2 = FastLanguageModel.get_peft_model(model.step1, r=32, ...)
model.step2.train(...)  # Train on domain-specific dataset

# Merge adaptations
final_model = model.step1.merge_and_unload() + model.step2.merge_and_unload()
```

### LoRA Ensembling

Combine multiple LoRAs for better performance:

```python
from peft import PeftModel

base_model = FastLanguageModel.from_pretrained("meta-llama/Llama-3.1-70B-hf")

# Load multiple LoRAs
lora1 = PeftModel.from_pretrained(base_model, "lora-adaptation-1")
lora2 = PeftModel.from_pretrained(base_model, "lora-adaptation-2")

# Ensemble predictions (simple averaging)
def ensemble_predict(text, models):
    outputs = [model(text) for model in models]
    return sum(outputs) / len(outputs)
```

### LoRA + QLoRA (2026 Standard)

```python
from unsloth import FastLanguageModel

# Load with 4-bit quantization + LoRA
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3.1-70B-hf",
    load_in_4bit=True,
    quantization_type="bnb",
)

# Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# Training with mixed precision
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        fp16=True,  # Mixed precision
        save_steps=100,
        logging_steps=10,
    )
)

trainer.train()
```

## Merging and Using LoRA Adaptations

### Merging LoRA into Base Model

```python
# Merge and unload LoRA
model = model.merge_and_unload()

# Save full model
model.save_pretrained("llama-3.1-70b-merged")
tokenizer.save_pretrained("llama-3.1-70b-merged")
```

### Loading a LoRA Adaptation

```python
from peft import PeftModel
from unsloth import FastLanguageModel

# Load base model
base_model, tokenizer = FastLanguageModel.from_pretrained(
    "meta-llama/Llama-3.1-70B-hf"
)

# Load LoRA adaptation
model = PeftModel.from_pretrained(base_model, "my-lora-adaptation")

# Switch between adaptations dynamically
model.load_adapter("another-lora-adaptation")
model.set_adapter("another-lora-adaptation")
```

### Inference with LoRA

```python
import torch

# Set to evaluation mode
model.eval()

# Generate with LoRA adaptation
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

## LoRA for Different Tasks

### Instruction Tuning

```python
# Dataset format
dataset = [
    {
        "instruction": "Explain the concept of machine learning",
        "input": "",
        "output": "Machine learning is a subset of artificial intelligence..."
    },
    # ... more examples
]

# Tokenize with special tokens
def tokenize_function(examples):
    return tokenizer(
        f"### Instruction: {examples['instruction']}\n\n### Input: {examples['input']}\n\n### Response: {examples['output']}",
        truncation=True,
        padding='max_length',
        max_length=2048
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)
```

### Role-Specific Finetuning

```python
# Medical LoRA
medical_lora = FastLanguageModel.get_peft_model(
    model,
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
)
medical_lora.train(medical_dataset)
medical_lora.save_pretrained("llama-medical-lora")

# Legal LoRA
legal_lora = FastLanguageModel.get_peft_model(
    model,
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
)
legal_lora.train(legal_dataset)
legal_lora.save_pretrained("llama-legal-lora")
```

### Dynamic LoRA Switching

```python
class LoRASwitcher:
    def __init__(self, base_model):
        self.base_model = base_model
        self.current_lora = None
        self.loras = {}

    def load_lora(self, name, path):
        lora = PeftModel.from_pretrained(self.base_model, path)
        self.loras[name] = lora

    def switch(self, name):
        if name not in self.loras:
            raise ValueError(f"LoRA {name} not loaded")
        self.current_lora = self.loras[name]

    def generate(self, prompt, **kwargs):
        if self.current_lora is None:
            raise ValueError("No LoRA selected")
        return self.current_lora.generate(prompt, **kwargs)

# Usage
switcher = LoRASwitcher(base_model)
switcher.load_lora("medical", "llama-medical-lora")
switcher.load_lora("legal", "llama-legal-lora")

# Switch between domains
switcher.switch("medical")
medical_response = switcher.generate("What causes diabetes?")

switcher.switch("legal")
legal_response = switcher.generate("Explain contract law.")
```

## LoRA Debugging and Monitoring

### Tracking LoRA Training

```python
import wandb
from torch.utils.tensorboard import SummaryWriter

# Initialize logging
wandb.init(project="lora-finetuning")
writer = SummaryWriter("logs")

# Training loop with logging
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for step, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        # Log metrics
        if step % 10 == 0:
            wandb.log({
                "loss": loss.item(),
                "epoch": epoch,
                "step": step
            })
            writer.add_scalar("Loss/train", loss.item(), epoch * len(train_dataloader) + step)

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}")
```

### Analyzing LoRA Weights

```python
import matplotlib.pyplot as plt
import numpy as np

# Extract LoRA weights
def get_lora_weights(model):
    weights = {}
    for name, module in model.named_modules():
        if "lora_A" in name:
            weights[name] = module.weight.data.cpu().numpy()
    return weights

# Plot weight distributions
lora_weights = get_lora_weights(model)
plt.figure(figsize=(12, 8))
for i, (name, weight) in enumerate(lora_weights.items()):
    plt.subplot(4, 4, i+1)
    plt.hist(weight.flatten(), bins=50, alpha=0.7)
    plt.title(name)
    plt.tight_layout()
plt.savefig("lora_weights.png")
```

### LoRA Weight Visualization

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Get all LoRA A matrices
all_A_weights = []
for name, module in model.named_modules():
    if "lora_A" in name:
        all_A_weights.append(module.weight.data.cpu().numpy().flatten())

# Apply PCA
all_A_weights = np.array(all_A_weights)
pca = PCA(n_components=2)
reduced = pca.fit_transform(all_A_weights)

# Plot
plt.figure(figsize=(10, 8))
plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7)
for i, name in enumerate(lora_weights.keys()):
    plt.annotate(name, (reduced[i, 0], reduced[i, 1]))
plt.title("LoRA Weight Space (PCA)")
plt.savefig("lora_pca.png")
```

## LoRA Optimization Techniques

### Learning Rate Scheduling

```python
from transformers import get_linear_schedule_with_warmup

# Calculate total training steps
total_steps = len(train_dataset) * num_epochs / batch_size

# Create scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0.1 * total_steps,
    num_training_steps=total_steps
)

# Add to training loop
for epoch in range(num_epochs):
    for step, batch in enumerate(train_dataloader):
        # ... forward, backward, step ...
        scheduler.step()
```

### Gradient Clipping

```python
from torch.nn.utils import clip_grad_norm_

# Add to training loop
for epoch in range(num_epochs):
    for step, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        # Clip gradients
        clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        optimizer.zero_grad()
```

### LoRA Dropout Tuning

```python
# Experiment with different dropout rates
for dropout_rate in [0.0, 0.05, 0.1, 0.2]:
    model = FastLanguageModel.get_peft_model(
        base_model,
        r=64,
        lora_dropout=dropout_rate,
        # ... other params ...
    )

    # Train and evaluate
    results = train_and_evaluate(model, train_dataset, val_dataset)
    print(f"Dropout {dropout_rate}: {results}")
```

## LoRA Best Practices (2026 Edition)

### 1. Start Simple
```python
# Baseline configuration
model = FastLanguageModel.get_peft_model(
    model,
    r=32,           # Start with moderate rank
    lora_alpha=64,  # Conservative alpha
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
```

### 2. Scale Appropriately
- **7B models**: r=16-64, batch_size=4-8
- **13B models**: r=32-128, batch_size=2-4
- **30B+ models**: r=64-256, batch_size=1-2

### 3. Monitor Overfitting
```python
from sklearn.metrics import accuracy_score

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return {"accuracy": accuracy_score(labels, predictions)}

# Use early stopping
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    args=TrainingArguments(
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        learning_rate=2e-4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        max_steps=1000,
        load_best_model_at_end=True,
    )
)
```

### 4. Use Mixed Precision
```python
# Enable automatic mixed precision
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    args=TrainingArguments(
        fp16=True,  # Mixed precision
        bf16=True,  # BF16 if available
        # ... other args ...
    )
)
```

### 5. Save Checkpoints Regularly
```python
# Save every N steps
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    args=TrainingArguments(
        save_steps=500,
        save_total_limit=5,
        output_dir="./lora_checkpoints",
    )
)
```

### 6. Document Your LoRA
```markdown
# LoRA Adaptation Card

- **Base Model**: Llama-3.1-70B
- **Rank (r)**: 64
- **Alpha**: 128
- **Dropout**: 0.05
- **Target Modules**: q_proj, k_proj, v_proj, o_proj
- **Training Data**: medical-qa-dataset-v1
- **Epochs**: 5
- **Batch Size**: 2
- **Learning Rate**: 2e-4
- **Date**: 2026-01-03
- **Performance**: 85% accuracy on validation set
- **Use Case**: Medical question answering
```

## LoRA vs Full Finetuning Comparison

| Metric | Full Finetuning | LoRA |
|--------|------------------|------|
| Trainable Parameters | 99% | <1% |
| Memory Usage | Very High | Low |
| Training Time | Slow | Fast |
| Hardware Required | Multiple GPUs | Single GPU |
| Model Size | Large | Small (base + LoRA) |
| Transferability | Specific task | Can be reused |
| Accuracy | High | Near-full (with proper tuning) |
| Cost | $$$$ | $ |

## When to Use LoRA

### ✅ Good for:
- Limited compute resources
- Quick experiments
- Domain adaptation
- Multiple task specializations
- Reproducible research
- Commercial applications (smaller model sizes)

### ❌ Not ideal for:
- Foundational model training
- Major architecture changes
- When you need to modify embedding layers significantly
- Extremely large scale training

## Future of LoRA (2026+)

Expected developments:
1. **Adaptive Rank LoRA**: Dynamically adjust rank per layer
2. **Multi-Modal LoRA**: Extend to vision-language models
3. **Distributed LoRA**: Train on multiple GPUs efficiently
4. **Quantized LoRA**: 8-bit LoRA for even better efficiency
5. **LoRA Distillation**: Distill multiple LoRAs into one
6. **Continuous LoRA**: Online learning with LoRA

## Resources

- [LoRA Paper (2021)](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper (2023)](https://arxiv.org/abs/2305.14314)
- [PEFT Library](https://github.com/huggingface/peft) - Hugging Face's LoRA implementation
- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [LoRA Examples](https://github.com/huggingface/peft/tree/main/examples)

## Troubleshooting

### Common Issues and Solutions

**Problem**: LoRA not learning
- **Check**: Learning rate too low (<1e-5)
- **Solution**: Increase to 2e-4 to 5e-4

**Problem**: High loss, no improvement
- **Check**: Target modules incorrect or missing
- **Solution**: Verify modules exist in your model

**Problem**: OOM errors
- **Check**: Batch size too large
- **Solution**: Reduce batch size or use gradient accumulation

**Problem**: Slow training
- **Check**: Gradient checkpointing disabled
- **Solution**: Enable `use_gradient_checkpointing="unsloth"`

**Problem**: LoRA weights exploding
- **Check**: No gradient clipping
- **Solution**: Add `clip_grad_norm_(model.parameters(), 1.0)`

**Problem**: Poor generalization
- **Check**: Overfitting on small dataset
- **Solution**: Add dropout, use early stopping, or get more data
