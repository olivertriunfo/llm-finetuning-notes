# Model Quantization for Efficient LLM Finetuning (2026 Edition)

Quantization reduces model size and memory usage while maintaining performance, making finetuning large models more accessible. In 2026, **4-bit quantization has been established as the optimal choice** for fixed model bits, achieving 98.9% accuracy recovery.

## What is Quantization?

Quantization is the process of reducing the precision of model weights and activations from 16-bit or 32-bit floating point to lower-bit representations (8-bit, 4-bit, etc.). This reduces:
- **Memory usage** (2-4x reduction for 8-bit, 4-5x for 4-bit)
- **GPU VRAM requirements** (critical for large models)
- **Training/inference speed** (faster computation, up to 5x with 4-bit)
- **Model file size** (easier to store and share)

## 2026 Quantization Guidelines

### Accuracy Recovery Rates (Validated)

Based on comprehensive evaluations across 500K+ model configurations:

- **8-bit Quantization**: **99.9% accuracy recovery** ✅ SAFE CHOICE
  - Minimal quality loss
  - Best for production when accuracy is critical
  - ~50% memory reduction vs FP16
  - Only use when you need maximum accuracy

- **4-bit Quantization**: **98.9% accuracy recovery** ✅ OPTIMAL CHOICE (2026)
  - Proven optimal for fixed model bits across all tested LLMs
  - ~75% memory reduction vs FP16
  - Recommended for most use cases
  - Industry-standard default

### Model Size Guidelines

Recommended precision format based on model size:

```python
QUANTIZATION_GUIDELINES = {
    # Micro models: 300M - 1B
    "300M-1B": {
        "recommended": "6-8 bit",
        "reason": "Smaller models maintain accuracy with higher precision",
        "memory_reduction": "25-50%",
        "examples": ["phi-2", "tiny-llama"]
    },

    # Small models: 3B - 7B
    "3B-7B": {
        "recommended": "4-bit (NF4)",
        "reason": "Sweet spot for production quality",
        "memory_reduction": "75%",
        "accuracy_recovery": "98.9%",
        "examples": ["llama-3.2-3b", "mistral-7b"]
    },

    # Medium models: 8B - 30B
    "8B-30B": {
        "recommended": "4-bit (NF4)",
        "reason": "Optimal for performance/quality trade-off",
        "memory_reduction": "75%",
        "accuracy_recovery": "98.9%",
        "examples": ["llama-3.1-8b", "qwen-30b"]
    },

    # Large models: 70B+
    "70B+": {
        "recommended": "4-bit (NF4) + LoRA",
        "reason": "Only practical way to finetune on single GPU",
        "memory_reduction": "75%",
        "accuracy_recovery": "98.9%",
        "examples": ["llama-2-70b", "mistral-large"]
    }
}

## Quantization Methods

### 1. Post-Training Quantization (PTQ)

Apply quantization after training without retraining:

```python
from transformers import AutoModelForCausalLM, AutoQuantizationConfig
from transformers.models.auto import model

# Create quantization config
quantization_config = AutoQuantizationConfig.bitsandbytes(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

# Load and quantize model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=quantization_config,
    device_map="auto"
)
```

### 2. Quantization-Aware Training (QAT)

Train with quantization in the loop:

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=quantization_config,
    device_map="auto"
)
```

### 3. QLoRA (Quantization-aware Low-Rank Adaptation)

The approach used by Unsloth - combine 4-bit quantization with LoRA:

```python
from unsloth import FastLanguageModel
import torch

# Load model with 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-2-7b-hf",
    max_seq_length=2048,
    load_in_4bit=True,
    quantization_type="bnb",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none"
)

# Training remains the same
# The base weights stay in 4-bit, only LoRA matrices are trained in 16-bit
```

## Quantization Types

### 4-bit Quantization

Most aggressive quantization, used in QLoRA:

```python
from bitsandbytes.nn import Linear4bit

# Different 4-bit quantizations
# NF4: NormalFloat4 - most common for LLMs
# FP4: FixedPoint4 - alternative approach

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True  # Apply second 8-bit quantization
)
```

### 8-bit Quantization

Good balance between size and accuracy:

```python
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False
)
```

### Dynamic Quantization

Quantize only activations, not weights:

```python
import torch.quantization

# Create quantization scheme
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)
```

## Quantization Parameters

### Key Configuration Options

```python
BnbConfig = {
    # Base quantization
    "load_in_4bit": True,  # Use 4-bit quantization
    "load_in_8bit": False,  # Use 8-bit quantization (mutually exclusive with 4-bit)

    # 4-bit specific
    "bnb_4bit_quant_type": "nf4",  # Quantization type: nf4 or fp4
    "bnb_4bit_compute_dtype": torch.bfloat16,  # Compute dtype: bf16 or fp16
    "bnb_4bit_use_double_quant": True,  # Enable double quantization

    # 8-bit specific
    "llm_int8_threshold": 6.0,  # Threshold for 8-bit quantization
    "llm_int8_has_fp16_weight": False,  # Store weights in fp16

    # General
    "bnb_4bit_block_size": 64,  # Block size for quantization (64 or 128)
    "bnb_4bit_symmetric": False,  # Use symmetric quantization
}
```

## Using Quantized Models

### Inference with Quantized Models

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from bitsandbytes.nn import Linear4bit, Linear8bitLt

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "your-quantized-model",
    load_in_8bit=True,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("your-quantized-model")

# Generate with quantized model
inputs = tokenizer("Explain quantum computing:", return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Finetuning Quantized Models

```python
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

# Load base model with quantization
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,
    device_map="auto"
)

# Apply LoRA
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    fp16=True,
    save_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    num_train_epochs=3,
    lr_scheduler_type="constant",
    warmup_ratio=0.1
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator
)

# Train
trainer.train()
```

## Evaluation of Quantized Models

Compare different quantization schemes:

```python
import evaluate

metrics = {
    "perplexity": evaluate.load("perplexity", module_type="metric"),
    "bleu": evaluate.load("bleu"),
}

models = [
    ("fp16", AutoModelForCausalLM.from_pretrained("model", torch_dtype=torch.float16")),
    ("int8", AutoModelForCausalLM.from_pretrained("model", load_in_8bit=True)),
    ("nf4", AutoModelForCausalLM.from_pretrained("model", load_in_4bit=True, bnb_4bit_quant_type="nf4")),
]

results = {}

for name, model in models:
    model.eval()

    # Calculate perplexity
    perplexity = metrics["perplexity"].compute(
        model_id=name,
        model=model,
        tokenizers=[tokenizer],
        add_start_token=False
    )

    results[name] = {
        "perplexity": perplexity["perplexity"],
        "memory": get_memory_usage(model),
        "speed": get_inference_speed(model, test_input)
    }

# Print comparison
import pandas as pd
pd.DataFrame(results).T
```

## Advanced Quantization Techniques

### Adaptive Quantization

Adjust quantization based on model layers:

```python
from transformers import AutoModelForCausalLM
import torch

class AdaptiveQuantizer:
    def __init__(self, model, threshold=6.0, min_scale=0.1):
        self.model = model
        self.threshold = threshold
        self.min_scale = min_scale

    def quantize(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                weights = module.weight.data
                abs_max = weights.abs().max()

                # Adaptive scale
                scale = max(abs_max / self.threshold, self.min_scale)

                # Quantize
                quantized = (weights / scale).round() * scale

                module.weight.data = quantized

        return self.model

# Usage
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
quantizer = AdaptiveQuantizer(model)
quantized_model = quantizer.quantize()
```

### SmoothQuant

Compress attention layers by redistributing weights:

```python
from transformers import AutoModelForCausalLM
import torch
import torch.nn.functional as F

def smooth_quant_layer(layer, alpha=0.5):
    """Apply SmoothQuant to a layer"""
    if not isinstance(layer, torch.nn.Linear):
        return layer

    # Save original weights
    W = layer.weight.data

    # Compute scaling factors
    input_scale = alpha / (alpha + F.softmax(W, dim=1))
    output_scale = (1 - alpha) / (1 - alpha + F.softmax(W, dim=0))

    # Scale weights
    W_scaled = W * input_scale * output_scale.transpose(0, 1)

    # Quantize scaled weights
    W_quant = torch.round(W_scaled / 0.01) * 0.01

    layer.weight.data = W_quant

    return layer

# Apply to model
for name, module in model.named_modules():
    if "self_attn" in name:
        model = smooth_quant_layer(module, alpha=0.5)
```

### GPTQ (Gradient-optimized PTQ)

High-quality post-training quantization:

```python
import torch
from transformers import AutoModelForCausalLM
from gptq import quantize, config

# Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# GPTQ config
quant_config = config.QuantConfig(
    bits=4,  # Quantize to 4 bits
    group_size=128,  # Group size for quantization
    fuse_layers=True,  # Fuse layers for efficiency
    sym=False  # Asymmetric quantization
)

# Quantize model
quantized_model = quantize(model, quant_config)

# Save quantized model
torch.save(quantized_model.state_dict(), "llama-2-7b-4bit-gptq.pt")
```

## 2026 Precision Formats

### NVFP4 (NVIDIA Blackwell)

NVIDIA's new precision format for inference on Blackwell GPUs:

```python
# NVFP4 is used automatically on Blackwell hardware
# Provides ~2x faster inference than FP16
model = AutoModelForCausalLM.from_pretrained(
    "your-model",
    device_map="auto",
    # NVFP4 selected automatically if on Blackwell GPU
    attn_implementation="flash_attention_2"
)

# Inference automatically uses NVFP4 on compatible hardware
outputs = model.generate(inputs, max_new_tokens=100)
```

### MXFP4 (OpenAI gpt-oss)

OpenAI's quantization format for their gpt-oss models:

```python
# Models come pre-quantized with MXFP4
model = AutoModelForCausalLM.from_pretrained(
    "openai/gpt-oss-20b-mxfp4",  # Pre-quantized
    device_map="auto"
)

# Already optimized with ~2x faster inference
outputs = model.generate(inputs, max_new_tokens=100)
```

### QALoRA (Quantization-Aware LoRA)

New variant that quantizes LoRA adapters during finetuning:

```python
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig

# QALoRA config: quantizes weights AND LoRA adapters
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    "model-name",
    quantization_config=quantization_config
)

# QALoRA: Also quantizes LoRA adapters
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    quantization_aware=True  # ✨ NEW for QALoRA
)

model = get_peft_model(model, lora_config)
```

## Quantization vs Accuracy Tradeoffs (2026 Edition)

| Quantization | Memory (7B) | Speed | Accuracy Recovery | Best For |
|-------------|------------|-------|------------------|----------|
| FP16 | ~14GB | 1.0x | 100% (baseline) | Maximum accuracy, training |
| INT8 | ~7GB | 1.5-2x | **99.9%** ✅ | Production with critical accuracy needs |
| NF4 (QLoRA) | ~3.5GB | 3-5x | **98.9%** ✅ OPTIMAL | Standard choice for most use cases |
| NVFP4 (Blackwell) | ~3.5GB | 5-10x | **98.9%** | Inference on Blackwell GPUs |
| MXFP4 (gpt-oss) | ~3.5GB | 5-10x | **98.9%** | Pre-quantized OpenAI models |
| FP4 | ~3.5GB | 3-5x | ~95% | Research, experimentation |

## Best Practices (2026)

### For Training:
1. **Use 4-bit NF4 as standard**: Proven 98.9% accuracy recovery, industry-standard default
2. **Apply QLoRA for parameter-efficient training**: Combine 4-bit quantization with LoRA
3. **Use double quantization**: Further reduces memory (NF4 + 8-bit)
4. **Keep activations in BF16**: Compute dtype should be bfloat16 for stability
5. **Use r=16-32 for LoRA**: 4-bit models work well with moderate rank
6. **Monitor accuracy**: Compare a few steps with FP16 baseline to validate
7. **Consider QALoRA for extreme efficiency**: Quantizes both weights and LoRA adapters

### For Inference:
1. **Use 4-bit NF4 by default**: Best balance of speed and accuracy (98.9% recovery)
2. **Use 8-bit only if accuracy is critical**: When you need 99.9% recovery
3. **Use NVFP4 on Blackwell GPUs**: Automatic 2x speedup over NF4
4. **Enable FlashAttention-2**: Works perfectly with all quantization formats
5. **Batch requests aggressively**: Quantization enables larger batch sizes
6. **Use continuous batching**: For serving applications with variable load

### For Deployment:
1. **Quantize before deployment**: Reduce model size
2. **Use onnx runtime**: For optimized inference
3. **Enable tensor parallelism**: For multi-GPU serving
4. **Monitor quantized performance**: Track accuracy in production
5. **Consider model fusion**: Combine with other optimizations

## Troubleshooting Quantization Issues

### Common Problems and Solutions

**Problem**: Poor accuracy with quantization
- **Solutions**:
  - Try different quantization types (nf4 vs fp4)
  - Use higher rank for LoRA (r=64 instead of r=16)
  - Check if you're using double quantization
  - Compare with baseline before quantization

**Problem**: OOM errors even with quantization
- **Solutions**:
  - Reduce batch size further
  - Use gradient accumulation
  - Freeze some layers
  - Use offloading to CPU

**Problem**: Training diverges with quantization
- **Solutions**:
  - Reduce learning rate
  - Increase warmup steps
  - Use more stable optimizer (AdamW with weight decay)
  - Check gradient norm

**Problem**: Slow training with quantization
- **Solutions**:
  - Use BF16 compute dtype instead of FP16
  - Enable FlashAttention
  - Use larger batch sizes (quantization helps with batch size)
  - Check if all layers are properly quantized

**Problem**: Different results between FP16 and quantized models
- **Solutions**:
  - Ensure same random seed
  - Use same tokenizer
  - Check if all parameters are identical
  - Verify quantization was applied correctly

## Resources

- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [GPTQ Paper](https://arxiv.org/abs/2210.11345)
- [BitsAndBytes Documentation](https://huggingface.co/docs/bitsandbytes/main/index)
- [SmoothQuant Paper](https://arxiv.org/abs/2211.10438)
- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [Hugging Face Quantization Guide](https://huggingface.co/docs/transformers/main/quantization)

## Comparison with Other Methods

| Method | Memory | Training Speed | Inference Speed | Accuracy | Ease of Use |
|--------|--------|----------------|------------------|-----------|-------------|
| Full FP16 | High | Baseline | Baseline | Best | Easy |
| LoRA (FP16) | Medium | Fast | Baseline | Good | Medium |
| QLoRA | Low | Very Fast | Fast | Good | Easy |
| INT8 | Medium | Medium | Very Fast | Good | Medium |
| GPTQ | Low | N/A | Fast | Good | Hard |
| Distillation | Medium | Medium | Fast | Medium | Hard |

## Future Directions

Emerging quantization techniques:
1. **Hybrid quantization**: Mix different bit widths per layer
2. **Dynamic bit-width**: Adjust precision based on context
3. **Neural quantization**: Learn optimal quantization
4. **Sparse-quantized models**: Combine sparsity and quantization
5. **Quantization-aware architecture search**: Find best quantization-aware models

## Case Study: QLoRA Finetuning

```python
from unsloth import FastLanguageModel
from datasets import load_dataset
import torch

# Step 1: Load quantized model
model, tokenizer = FastLanguageModel.from_pretrained(
    "meta-llama/Llama-2-70B-hf",
    max_seq_length=2048,
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# Step 2: Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=64,  # Higher rank for better capacity
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth"
)

# Step 3: Load and prepare dataset
dataset = load_dataset("my-dataset", split="train")

def tokenize_function(examples):
    return tokenizer(
        f"### Instruction: {examples['instruction']}\n\n### Input: {examples['input']}\n\n### Response: {examples['output']}",
        truncation=True,
        padding='max_length',
        max_length=2048
    )

train_dataset = dataset.map(tokenize_function, batched=True)

# Step 4: Training setup
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0.1 * len(train_dataset),
    num_training_steps=len(train_dataset) * 3
)

# Step 5: Training loop
for epoch in range(3):
    model.train()
    for step, batch in enumerate(train_dataset):
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if step % 100 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")

# Step 6: Save results
model.save_pretrained("llama-2-70b-qlora-finetuned")
tokenizer.save_pretrained("llama-2-70b-qlora-finetuned")
```

This approach allows finetuning a 70B model on a single GPU with minimal accuracy loss!