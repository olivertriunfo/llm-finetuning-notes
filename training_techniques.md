# Advanced Training Techniques for LLM Finetuning

This document covers advanced techniques to improve the efficiency, effectiveness, and quality of your LLM finetuning.

## 1. Learning Rate Scheduling

Proper learning rate scheduling is crucial for effective training.

### Common Schedulers

#### Linear Warmup with Decay

```python
from transformers import get_linear_schedule_with_warmup

# Calculate total training steps
total_steps = len(train_dataset) * num_epochs // batch_size

# Create scheduler with warmup
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0.1 * total_steps,  # 10% of training for warmup
    num_training_steps=total_steps
)

# Add to training loop
for epoch in range(num_epochs):
    for step, batch in enumerate(train_dataloader):
        # Forward pass, backward pass
        loss.backward()

        # Update learning rate
        scheduler.step()

        # Update weights
        optimizer.step()
        optimizer.zero_grad()
```

#### Cosine Scheduling

```python
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0.05 * total_steps,
    num_training_steps=total_steps,
    num_cycles=0.5  # Half cosine cycle
)
```

#### Constant with Warmup

```python
from transformers import get_constant_schedule_with_warmup

scheduler = get_constant_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0.1 * total_steps
)
```

### Learning Rate Finding

```python
import torch
import numpy as np

def lr_find(model, train_loader, init_lr=1e-7, final_lr=1, beta=0.98):
    """Learning rate finder"""
    lr_mult = (final_lr / init_lr) ** (1.0 / len(train_loader))
    lrs = [init_lr * lr_mult**i for i in range(len(train_loader))]

    avg_loss = 0
    losses = []
    model.train()

    for i, (inputs, targets) in enumerate(train_loader):
        lr = lrs[i]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Smooth loss
        avg_loss = beta * avg_loss + (1-beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta**(i+1))
        losses.append(smoothed_loss)

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return lrs, losses

# Usage
lrs, losses = lr_find(model, train_dataloader)

# Plot to find optimal LR
import matplotlib.pyplot as plt
plt.plot(lrs, losses)
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.title('Learning Rate Finder')
plt.savefig('lr_find.png')
```

## 2. Gradient Checkpointing

Reduce memory usage by trading compute for memory:

### Standard Gradient Checkpointing

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model.gradient_checkpointing_enable()

# For PEFT models
model.model.gradient_checkpointing_enable()
```

### Unsloth-Specific Checkpointing

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained("meta-llama/Llama-2-7b-hf")

# Enable Unsloth's advanced checkpointing
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    use_gradient_checkpointing="unsloth",  # Use Unsloth's checkpointing
    # ... other params
)
```

## 3. Mixed Precision Training

Train with lower precision for speed and memory savings:

### FP16 Training

```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

for epoch in range(num_epochs):
    for batch in train_dataloader:
        optimizer.zero_grad()

        with autocast():
            outputs = model(**batch)
            loss = outputs.loss

        # Scale loss and backprop
        scaler.scale(loss).backward()

        # Unscale gradients and clip
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update weights
        scaler.step(optimizer)
        scaler.update()
```

### BF16 Training (Better for some LLMs)

```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler(init_scale=65536, mode='bf16')

for batch in train_dataloader:
    optimizer.zero_grad()

    with autocast(dtype=torch.bfloat16):
        outputs = model(**batch)
        loss = outputs.loss

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## 4. Gradient Accumulation

Simulate larger batch sizes by accumulating gradients:

```python
import torch

accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(train_dataloader):
    outputs = model(**batch)
    loss = outputs.loss / accumulation_steps  # Scale loss
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# Alternative: Gradient accumulation with scaler
scaler = GradScaler()
optimizer.zero_grad()

for i, batch in enumerate(train_dataloader):
    with autocast():
        outputs = model(**batch)
        loss = outputs.loss / accumulation_steps

    scaler.scale(loss).backward()

    if (i + 1) % accumulation_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

## 5. Early Stopping

Stop training when validation performance plateaus:

```python
import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience=5, delta=0, save_path="checkpoint.pt"):
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss

# Usage
early_stopping = EarlyStopping(patience=3, save_path="/workspace/best_model.pt")

for epoch in range(num_epochs):
    # Training
    train_loss = train_one_epoch(model, train_loader, optimizer)

    # Validation
    val_loss = validate(model, val_loader)

    # Early stopping
    early_stopping(val_loss, model)

    if early_stopping.early_stop:
        print("Early stopping triggered")
        break
```

## 6. Checkpointing and Resuming

Save and resume training progress:

### Save Checkpoints

```python
import os
import torch
from datetime import datetime

def save_checkpoint(model, optimizer, scheduler, epoch, step, save_dir="/workspace/checkpoints"):
    """Save training checkpoint"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'timestamp': datetime.now().isoformat()
    }

    checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}_step_{step}.pt")
    torch.save(checkpoint, checkpoint_path)

    # Save latest checkpoint
    torch.save(checkpoint, os.path.join(save_dir, "latest_checkpoint.pt"))

    return checkpoint_path
```

### Load Checkpoints

```python
def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load training checkpoint"""
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint['epoch']
    step = checkpoint['step']

    print(f"Resumed from epoch {epoch}, step {step}")
    return epoch, step

# Usage
checkpoint_path = "/workspace/checkpoints/latest_checkpoint.pt"

if os.path.exists(checkpoint_path):
    start_epoch, start_step = load_checkpoint(model, optimizer, scheduler, checkpoint_path)
else:
    start_epoch, start_step = 0, 0

# Continue training from where you left off
for epoch in range(start_epoch, num_epochs):
    for step in range(start_step if epoch == start_epoch else 0, len(train_loader)):
        # Training code

        if step % 100 == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, step)
```

## 7. Gradient Clipping

Prevent exploding gradients:

```python
import torch

# Method 1: Clip by norm
max_norm = 1.0
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

# Method 2: Clip by value
max_value = 1.0
torch.nn.utils.clip_grad_value_(model.parameters(), max_value)

# Add to training loop
for batch in train_dataloader:
    optimizer.zero_grad()

    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()

    # Clip gradients before step
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()
```

## 8. Weight Decay and Regularization

Prevent overfitting:

```python
import torch

# Method 1: L2 Regularization (built into AdamW)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-4,
    weight_decay=0.01  # L2 regularization strength
)

# Method 2: Dropout
from transformers import AutoConfig

config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
config.hidden_dropout_prob = 0.1  # Add dropout
config.attention_probs_dropout_prob = 0.1

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    config=config
)

# Method 3: Stochastic Depth (for some architectures)
# Method 4: Label Smoothing
```

## 9. Batch Normalization Alternatives

For LLMs, LayerNorm is typically used instead of BatchNorm:

```python
# Llama uses RMSNorm or LayerNorm
from transformers.models.llama.configuration_llama import LlamaConfig

config = LlamaConfig(
    # ... other params
    rms_norm_eps=1e-6,  # For RMSNorm
    hidden_dropout=0.0  # Typically no dropout in pretrained models
)
```

## 10. Distributed Training

### Single-Machine Multi-GPU

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Initialize distributed training
def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# Create model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = DDP(model, device_ids=[rank])

# Create sampler
sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)

# Training loop
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)
    for batch in train_dataloader:
        # Training code
        pass

# Cleanup
dist.destroy_process_group()
```

### Multi-Node Training

```python
# Use torchrun to launch training
# torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr="192.168.1.1" --master_port=1234 train.py

import os
import torch
import torch.distributed as dist

# Get distributed training parameters
local_rank = int(os.environ["LOCAL_RANK"])
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

# Initialize distributed training
dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
torch.cuda.set_device(local_rank)

# Create model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf").to(local_rank)
model = DDP(model, device_ids=[local_rank])

# ... rest of training code
```

### FSDP (Fully Sharded Data Parallel)

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import auto_wrap_policy

# Create model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Wrap layers
auto_wrap_policy = auto_wrap_policy(model, min_num_params=10**6)

# Create FSDP model
model = FSDP(model, auto_wrap_policy=auto_wrap_policy, device_id=torch.cuda.current_device())

# ... training code
```

## 11. Adaptive Training Techniques

### Learning Rate Free (LRF) Optimization

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    # ... other args
    lr_scheduler_type="constant_with_warmup",
    learning_rate=1e-4,
    warmup_ratio=0.1,
)
```

### Cosine with Restarts

```python
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=100,  # Number of iterations for the first restart
    T_mult=2,  # A factor increases T_i after a restart
    eta_min=1e-6  # Minimum learning rate
)
```

## 12. Memory Optimization

### Memory-Efficient Data Loading

```python
from torch.utils.data import DataLoader

# Use pin_memory and num_workers
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=4,  # Number of worker processes
    pin_memory=True,  # Faster data transfer to GPU
    shuffle=True,
    drop_last=True
)
```

### Gradient Checkpointing with Offloading

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": True},
    fp16=True,
    offload_state_dict=True,  # Offload model weights to CPU
    offload_optimizer_state_dict=True,  # Offload optimizer to CPU
)
```

## 13. Training with FlashAttention

```python
from transformers import LlamaConfig

config = LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
config._attn_implementation = "flash_attention_2"  # Enable FlashAttention

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    config=config
)
```

## 14. Training Monitoring and Logging

### WandB Integration

```python
import wandb

# Initialize WandB
wandb.init(
    project="llm-finetuning",
    name=f"llama-2-7b-lora-lr{learning_rate}",
    config={
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "model": "llama-2-7b",
        "method": "QLoRA"
    }
)

# Log training metrics
for epoch in range(num_epochs):
    for step, batch in enumerate(train_dataloader):
        loss = train_step(batch)

        if step % 10 == 0:
            wandb.log({
                "loss": loss,
                "epoch": epoch,
                "step": step,
                "lr": scheduler.get_last_lr()[0]
            })

# Log model artifacts
wandb.save("checkpoint.pt")
wandb.finish()
```

### TensorBoard Integration

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/llama-2-7b-finetuning")

for epoch in range(num_epochs):
    for step, batch in enumerate(train_dataloader):
        loss = train_step(batch)

        if step % 10 == 0:
            writer.add_scalar("Loss/train", loss, epoch * len(train_dataloader) + step)
            writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], epoch * len(train_dataloader) + step)

writer.close()
```

## Best Practices

1. **Start small**: Test with small batches and fewer epochs first
2. **Monitor memory**: Use `nvidia-smi` to track GPU usage
3. **Use mixed precision**: Always use FP16 or BF16 when possible
4. **Checkpoint frequently**: Save every 500-1000 steps
5. **Monitor gradients**: Check for exploding/vanishing gradients
6. **Validate regularly**: Check validation metrics every epoch
7. **Use learning rate finder**: Find optimal learning rate before full training
8. **Warmup is crucial**: Always use learning rate warmup
9. **Balance batch size**: Larger batches need more memory but train faster
10. **Log everything**: Track all hyperparameters and metrics

## Resources

- [Hugging Face Training Documentation](https://huggingface.co/docs/transformers/training)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [FairScale Documentation](https://facebookresearch.github.io/fairscale/)
- [PyTorch Distributed Training](https://pytorch.org/docs/stable/distributed.html)
- [FlashAttention Repository](https://github.com/Dao-AILab/flash-attention)
- [WandB Documentation](https://docs.wandb.ai/)
- [TensorBoard Documentation](https://www.tensorflow.org/tensorboard)

## Troubleshooting

### Common Issues and Solutions

**Problem**: OOM errors
- **Solution**: Reduce batch size, use gradient accumulation, enable gradient checkpointing

**Problem**: Training is too slow
- **Solution**: Use mixed precision, enable FlashAttention, increase batch size

**Problem**: Loss doesn't decrease
- **Solution**: Check learning rate (too high or too low), verify data loading, check for NaN values

**Problem**: NaN loss
- **Solution**: Clip gradients, check for data issues, reduce learning rate

**Problem**: Memory leaks
- **Solution**: Ensure proper cleanup, avoid global variables that accumulate, monitor memory usage

**Problem**: Gradient explosion
- **Solution**: Use gradient clipping, reduce learning rate, add regularization

**Problem**: Gradient vanishing
- **Solution**: Use proper initialization, consider residual connections, increase learning rate

**Problem**: Overfitting
- **Solution**: Add dropout, use weight decay, get more data, early stopping

**Problem**: Underfitting
- **Solution**: Train longer, increase model capacity, use higher learning rate, check data quality