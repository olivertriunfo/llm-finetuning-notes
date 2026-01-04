# C-LoRA: Continual Low-Rank Adaptation

C-LoRA (Continual Low-Rank Adaptation) extends the LoRA technique to support **continual learning** - the ability to learn multiple tasks sequentially without catastrophic forgetting.

## What is C-LoRA?

C-LoRA addresses the challenge of **sequential task learning** in LLMs. Traditional finetuning suffers from:
- **Catastrophic Forgetting**: Learning a new task erases knowledge of previous tasks
- **Task Interference**: Parameters optimized for one task degrade performance on others
- **Storage Overhead**: Need to store separate models for each task

C-LoRA solves these problems by:
1. Maintaining a **base model** (frozen)
2. Adding **task-specific LoRA layers** for each new task
3. Using **adaptation gates** to control knowledge transfer between tasks
4. Supporting **dynamic task switching** at inference time

## Why C-LoRA?

### Key Advantages

- **Single Model, Multiple Tasks**: Store one base model + small LoRA adaptations
- **No Forgetting**: Previous tasks remain unaffected when learning new ones
- **Dynamic Switching**: Change tasks at inference time
- **Memory Efficient**: Only task-specific parameters are stored
- **Scalable**: Add as many tasks as needed

### Use Cases

1. **Multi-Task Assistants**: Medical, legal, technical support in one model
2. **Domain Adaptation**: Switch between industries (finance, healthcare, etc.)
3. **User Personalization**: Different adaptations for different user groups
4. **Temporal Adaptation**: Update model with new information over time
5. **Privacy-Preserving**: Task-specific knowledge stays isolated

## How C-LoRA Works

### Key Insight: Can One LoRA Replace Multiple LoRAs?

The core question addressed by C-LoRA is: "Can one LoRA adapter replace multiple LoRAs?" The answer lies in the **learnable routing matrix** (𝓡) that dynamically controls how parameter subspaces are activated and updated. This matrix's element magnitudes influence the degree of forgetting - excessive updates to shared subspaces risk overwriting prior knowledge. C-LoRA's architecture is specifically designed to mitigate this interference through orthogonality constraints.

### Addressing Catastrophic Forgetting

Catastrophic forgetting is a fundamental challenge in continual learning where learning new tasks causes significant performance degradation on previously learned tasks. The paper identifies two key mechanisms of forgetting:

1. **Routing Matrix Interference**: The learnable routing matrix 𝓡 dynamically controls parameter subspace activation. When new tasks update this matrix, they can interfere with the routing paths established for previous tasks.

2. **LoRA Subspace Overlap**: Low-rank adaptation subspaces can overlap between tasks, causing new task updates to overwrite or interfere with the knowledge encoded in previous task adapters.

C-LoRA mitigates these issues through:
- **Orthogonality Constraints**: Ensures that parameter subspaces for different tasks remain orthogonal, preventing interference.
- **Shared Low-Rank Subspaces**: Leverages common knowledge across tasks while maintaining task-specific adaptations.
- **Adaptation Gates**: Controls the strength of task-specific updates, allowing the model to preserve base knowledge when appropriate.

The paper's experiments demonstrate that C-LoRA outperforms existing state-of-the-art continual learning methods across multiple datasets, showing minimal forgetting while maintaining high performance on new tasks.

### Architecture

```
┌───────────────────────────────────────────────────┐
│              Base Model (Frozen)                  │
└───────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────┐
│                Task 1 Adapter                     │
│  ┌─────────────┐    ┌─────────────────────────┐   │
│  │  LoRA B₁    │    │  LoRA A₁               │   │
│  └─────────────┘    └─────────────────────────┘   │
│  ┌─────────────┐    ┌─────────────────────────┐   │
│  │  Gate G₁    │    │  Scaling α₁            │   │
│  └─────────────┘    └─────────────────────────┘   │
└───────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────┐
│                Task 2 Adapter                     │
│  ┌─────────────┐    ┌─────────────────────────┐   │
│  │  LoRA B₂    │    │  LoRA A₂               │   │
│  └─────────────┘    └─────────────────────────┘   │
│  ┌─────────────┐    ┌─────────────────────────┐   │
│  │  Gate G₂    │    │  Scaling α₂            │   │
│  └─────────────┘    └─────────────────────────┘   │
└───────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────┐
│                Task N Adapter                     │
└───────────────────────────────────────────────────┘
```

### Mathematical Formulation

For each layer with weight matrix W₀:

```
W_task = W₀ + (G_task × (B_task × A_task) × α_task)
```

Where:
- `W₀`: Frozen base model weights
- `B_task`, `A_task`: Task-specific LoRA matrices
- `G_task`: Task gate controlling adaptation strength
- `α_task`: Task-specific scaling factor

The **gate mechanism** is key:
- `G_task ≈ 0`: Use base model knowledge (preserve previous tasks)
- `G_task ≈ 1`: Use task-specific adaptation (learn new task)

## Implementing C-LoRA

### Option 1: Using PEFT Library

```python
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load base model (frozen)
model_name = "meta-llama/Llama-2-7b-hf"
base_model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Freeze the base model
for param in base_model.parameters():
    param.requires_grad = False

# Task 1: Medical QA
medical_config = LoraConfig(
    r=16,                        # Rank
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    inference_mode=False,
)

medical_model = get_peft_model(base_model, medical_config)

# Train on medical dataset
medical_dataset = load_medical_dataset()
train(medical_model, medical_dataset)

# Save medical adaptation
medical_model.save_pretrained("llama-2-7b-medical-lora")

# Task 2: Legal Assistant
legal_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    inference_mode=False,
)

legal_model = get_peft_model(base_model, legal_config)

# Train on legal dataset
legal_dataset = load_legal_dataset()
train(legal_model, legal_dataset)

# Save legal adaptation
legal_model.save_pretrained("llama-2-7b-legal-lora")
```

### Option 2: Custom C-LoRA Implementation

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class CLoRAAdapter(nn.Module):
    """Continual LoRA Adapter"""
    def __init__(self, base_model, rank=16, alpha=16, modules=None):
        super().__init__()
        self.base_model = base_model
        self.rank = rank
        self.alpha = alpha

        # Store task-specific adapters
        self.adapters = nn.ModuleDict()

        if modules is None:
            modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        self.modules_to_adapt = modules

    def add_task(self, task_name):
        """Add a new task adapter"""
        adapter = nn.ModuleDict()

        for name, module in self.base_model.named_modules():
            if any(target in name for target in self.modules_to_adapt):
                for param_name, param in module.named_parameters():
                    if param.requires_grad:
                        # Create LoRA matrices
                        lora_A = nn.Linear(param.size(1), self.rank, bias=False)
                        lora_B = nn.Linear(self.rank, param.size(0), bias=False)

                        # Create gate
                        gate = nn.Parameter(torch.ones(1))

                        adapter[f"{name}.{param_name}.A"] = lora_A
                        adapter[f"{name}.{param_name}.B"] = lora_B
                        adapter[f"{name}.{param_name}.gate"] = gate

        self.adapters[task_name] = adapter
        return task_name

    def forward(self, task_name, *args, **kwargs):
        """Forward pass with task-specific adapter"""
        if task_name not in self.adapters:
            raise ValueError(f"Task {task_name} not found")

        # Forward through base model
        outputs = self.base_model(*args, **kwargs)

        # Apply task-specific adaptations
        for name, module in self.base_model.named_modules():
            if any(target in name for target in self.modules_to_adapt):
                for param_name, param in module.named_parameters():
                    if param.requires_grad:
                        # Get LoRA components
                        lora_A = self.adapters[task_name][f"{name}.{param_name}.A"]
                        lora_B = self.adapters[task_name][f"{name}.{param_name}.B"]
                        gate = self.adapters[task_name][f"{name}.{param_name}.gate"]

                        # Compute LoRA update
                        delta = lora_B(lora_A(outputs[0])) * self.alpha
                        delta = delta * torch.sigmoid(gate)

                        # Modify output
                        # (Implementation depends on exact layer type)

        return outputs

# Usage
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
clora = CLoRAAdapter(base_model, rank=16, alpha=16)

# Add tasks
task1 = clora.add_task("medical")
task2 = clora.add_task("legal")

# Train each task
# ... training code ...

# Inference with task switching
medical_output = clora.forward("medical", inputs=medical_prompt)
legal_output = clora.forward("legal", inputs=legal_prompt)
```

### Option 3: Using Unsloth with C-LoRA

```python
from unsloth import FastLanguageModel
from peft import PeftModel
import torch

# Load base model with Unsloth
base_model, tokenizer = FastLanguageModel.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    max_seq_length=2048,
)

# Store base model weights (will be frozen)
base_model.save_pretrained("base_model")

# Task 1: Train medical adaptation
medical_model = FastLanguageModel.get_peft_model(
    base_model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
)

medical_model.train(medical_dataset)
medical_model.save_pretrained("llama-2-7b-medical")

# Task 2: Load base and train legal adaptation
# Important: Always load from the same base model
base_model = FastLanguageModel.from_pretrained("base_model")

legal_model = FastLanguageModel.get_peft_model(
    base_model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
)

legal_model.train(legal_dataset)
legal_model.save_pretrained("llama-2-7b-legal")

# Inference with task switching
def generate(task_name, prompt):
    base = FastLanguageModel.from_pretrained("base_model")
    adapter = PeftModel.from_pretrained(base, f"llama-2-7b-{task_name}")

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = adapter.generate(**inputs, max_new_tokens=100)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Use different tasks
medical_response = generate("medical", "What causes diabetes?")
legal_response = generate("legal", "Explain contract law.")
```

## C-LoRA Training Strategies

### 1. Sequential Task Learning

```python
# Pseudocode for sequential learning
base_model = load_base_model()

for task in tasks:
    # Load task dataset
    dataset = load_dataset(task)

    # Create fresh LoRA adapter for this task
    adapter = create_lora_adapter(base_model)

    # Train only this adapter
    train(adapter, dataset)

    # Save task-specific adapter
    save_adapter(adapter, task)

    # Optional: Evaluate on all previous tasks
    evaluate(adapter, all_previous_tasks)
```

### 2. Rehearsal Learning

```python
# Include small samples from previous tasks
current_task_data = load_current_task_dataset()

# Sample from previous tasks (rehearsal)
rehearsal_data = sample_from_previous_tasks(percentage=0.1)

# Combine
combined_dataset = current_task_data + rehearsal_data

# Train with combined dataset
adapter = create_lora_adapter(base_model)
train(adapter, combined_dataset)
```

### 3. Regularization Techniques

```python
# Elastic Weight Consolidation (EWC)
def add_ewc_regularization(model, previous_task_params, fisher_info):
    """Add EWC regularization to preserve previous task knowledge"""
    def ewc_loss(current_params):
        loss = 0
        for (name, param), (prev_param, fisher) in zip(
            current_params,
            zip(previous_task_params.values(), fisher_info.values())
        ):
            loss += (fisher * (param - prev_param).pow(2)).sum()
        return loss

    return ewc_loss

# Usage
previous_params = {n: p for n, p in model.named_parameters()}
fisher_info = estimate_fisher_information(model, validation_data)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for batch in dataset:
    # Forward pass
    outputs = model(batch)
    task_loss = compute_loss(outputs, batch)

    # EWC regularization
    ewc_loss = add_ewc_regularization(
        model.parameters(),
        previous_params,
        fisher_info
    )

    # Total loss
    total_loss = task_loss + 0.1 * ewc_loss

    # Backward
    total_loss.backward()
    optimizer.step()
```

### 4. Gradient Episodic Memory (GEM)

```python
class GEM:
    def __init__(self, memory_size=20):
        self.memory_size = memory_size
        self.memory = {}

    def update_memory(self, task_name, gradients):
        """Store gradients for this task"""
        if task_name not in self.memory or len(self.memory[task_name]) < self.memory_size:
            self.memory[task_name] = gradients
        else:
            # Replace oldest gradient
            self.memory[task_name].pop(0)
            self.memory[task_name].append(gradients)

    def constrain_gradients(self, current_gradients):
        """Constrain current gradients to not interfere with past tasks"""
        for task_grads in self.memory.values():
            constraint = project(current_gradients, task_grads)
            current_gradients = constraint
        return current_gradients

# Usage
gem = GEM(memory_size=20)

for task in tasks:
    for batch in task_dataset:
        # Compute gradients
        gradients = compute_gradients(model, batch)

        # Constrain gradients
        constrained_grads = gem.constrain_gradients(gradients)

        # Apply constrained gradients
        apply_gradients(model, constrained_grads)

    # Store gradients for this task
    gem.update_memory(task, gradients)
```

## Inference with Task Switching

### Dynamic Task Selection

```python
from peft import PeftModel
import torch

class CLoRAInference:
    def __init__(self, base_model_path, task_paths):
        self.base_model_path = base_model_path
        self.task_paths = task_paths
        self.models = {}

    def load_task(self, task_name):
        """Load task-specific adapter"""
        if task_name not in self.models:
            base_model = FastLanguageModel.from_pretrained(self.base_model_path)
            adapter = PeftModel.from_pretrained(base_model, self.task_paths[task_name])
            self.models[task_name] = adapter
        return self.models[task_name]

    def generate(self, task_name, prompt, **kwargs):
        """Generate with task-specific adapter"""
        model = self.load_task(task_name)
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(**inputs, **kwargs)

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def batch_generate(self, requests):
        """Handle multiple requests with different tasks"""
        results = {}

        # Group by task
        task_groups = {}
        for req_id, (task, prompt) in requests.items():
            if task not in task_groups:
                task_groups[task] = []
            task_groups[task].append((req_id, prompt))

        # Process each task group
        for task, group in task_groups.items():
            model = self.load_task(task)

            for req_id, prompt in group:
                inputs = tokenizer(prompt, return_tensors="pt")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=100)

                results[req_id] = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return results

# Usage
clora = CLoRAInference(
    base_model_path="base_model",
    task_paths={
        "medical": "llama-2-7b-medical",
        "legal": "llama-2-7b-legal",
        "coding": "llama-2-7b-coding"
    }
)

# Single request
response = clora.generate("medical", "What is hypertension?")

# Multiple requests with different tasks
batch_requests = {
    1: ("medical", "Explain diabetes"),
    2: ("legal", "What is a contract?"),
    3: ("medical", "Symptoms of flu")
}
batch_results = clora.batch_generate(batch_requests)
```

### Zero-Shot Task Identification

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class TaskClassifier:
    def __init__(self, task_descriptions):
        self.vectorizer = TfidfVectorizer()
        self.task_descriptions = task_descriptions
        self.task_vectors = self.vectorizer.fit_transform(task_descriptions)

    def classify_task(self, prompt):
        """Classify prompt to most similar task"""
        prompt_vec = self.vectorizer.transform([prompt])
        similarities = cosine_similarity(prompt_vec, self.task_vectors)[0]
        return np.argmax(similarities)

# Usage
task_descriptions = {
    "medical": "medical questions about diseases, symptoms, treatments, anatomy, pharmacology",
    "legal": "legal questions about laws, contracts, regulations, court cases, legal rights",
    "coding": "programming questions about code, algorithms, software development, debugging"
}

classifier = TaskClassifier(task_descriptions)

# Classify prompt automatically
task = classifier.classify_task("What causes high blood pressure?")
print(f"Detected task: {task}")  # Output: medical

# Use detected task
response = clora.generate(task, "What causes high blood pressure?")
```

## Evaluating C-LoRA Performance

### Metrics to Track

1. **Forward Transfer**: Improvement on new task
2. **Backward Transfer**: Impact on previous tasks
3. **Memory Efficiency**: Storage overhead
4. **Inference Latency**: Task switching cost

```python
import pandas as pd

class CLORAEvaluator:
    def __init__(self, base_model, task_adapters):
        self.base_model = base_model
        self.task_adapters = task_adapters
        self.results = {}

    def evaluate_task(self, task_name, test_datasets):
        """Evaluate performance on task and all previous tasks"""
        results = {}

        # Load adapter
        adapter = PeftModel.from_pretrained(self.base_model, self.task_adapters[task_name])

        # Evaluate on current task
        current_perf = self._evaluate(adapter, test_datasets[task_name])
        results[f"{task_name}_current"] = current_perf

        # Evaluate on all previous tasks
        for prev_task in self.task_adapters:
            if prev_task == task_name:
                continue

            # Load previous task adapter
            prev_adapter = PeftModel.from_pretrained(
                self.base_model,
                self.task_adapters[prev_task]
            )

            # Evaluate on previous task's data
            prev_perf = self._evaluate(prev_adapter, test_datasets[task_name])
            results[f"{prev_task}_backward"] = prev_perf

        return results

    def _evaluate(self, model, dataset):
        """Compute accuracy/metrics"""
        # Implementation depends on your task
        correct = 0
        total = 0

        for sample in dataset:
            inputs = tokenizer(sample["input"], return_tensors="pt")
            labels = sample["output"]

            with torch.no_grad():
                outputs = model.generate(**inputs)

            # Compare outputs to labels
            # ... metric computation ...
            total += 1
            correct += 1 if is_correct(outputs, labels) else 0

        return correct / total

    def plot_performance(self):
        """Plot forward and backward transfer"""
        df = pd.DataFrame(self.results).T
        df.plot(kind='bar')
        # ... plotting code ...

# Usage
evaluator = CLORAEvaluator(
    base_model="base_model",
    task_adapters={
        "task1": "adapter1",
        "task2": "adapter2",
        "task3": "adapter3"
    }
)

results = {}
for task in ["task1", "task2", "task3"]:
    task_results = evaluator.evaluate_task(task, test_datasets)
    results[task] = task_results

    # Plot cumulative performance
    evaluator.plot_performance()
```

### Catastrophic Forgetting Metric

```python
def compute_forgetting(task_performance):
    """
    Compute catastrophic forgetting metric

    Args:
        task_performance: Dictionary of {task: [acc1, acc2, ..., accN]}
        where acc1 is performance after task 1, acc2 after task 2, etc.

    Returns:
        Forgetting score (lower is better)
    """
    total_forgetting = 0
    num_comparisons = 0

    for task, accs in task_performance.items():
        max_acc = max(accs)  # Best performance on this task

        for acc in accs[1:]:  # Compare to later evaluations
            total_forgetting += max(0, max_acc - acc)  # Only count decreases
            num_comparisons += 1

    return total_forgetting / num_comparisons if num_comparisons > 0 else 0

# Example usage
task_performance = {
    "medical": [0.85, 0.84, 0.83],  # Medical acc after each task
    "legal": [0.78, 0.77, 0.76],    # Legal acc after each task
    "coding": [0.88, 0.87, 0.86],   # Coding acc after each task
}

forgetting_score = compute_forgetting(task_performance)
print(f"Catastrophic Forgetting Score: {forgetting_score:.4f}")
```

## C-LoRA vs Other Approaches

### Comparison Table

| Approach | Forgetting | Storage | Inference Speed | Task Switching | Implementation Complexity |
|----------|-----------|---------|-----------------|----------------|---------------------------|
| **C-LoRA** | Minimal | Low (per-task) | Fast | Instant | Medium |
| Full Finetuning | High | High (per-task) | Slow | Slow | Low |
| Standard LoRA | Medium | Low (per-task) | Fast | Instant | Low |
| EWC | Low | Medium | Fast | Instant | High |
| GEM | Low | Medium | Fast | Instant | High |
| Progressive Networks | None | Very High | Slow | Fast | High |
| Replay Buffer | Low | Very High | Slow | Instant | Medium |

### When to Use C-LoRA

**✅ Good for:**
- Multiple related tasks
- Limited GPU memory
- Need to preserve all task knowledge
- Dynamic task selection at inference
- Production systems with evolving requirements

**❌ Not ideal for:**
- Completely unrelated tasks (e.g., image + text)
- Tasks requiring very different model architectures
- When you can afford full model duplication

## Advanced C-LoRA Techniques

### 1. Hierarchical Task Organization

```python
class HierarchicalCLoRA:
    def __init__(self, base_model):
        self.base_model = base_model
        self.task_hierarchy = {}

    def add_task_group(self, group_name, parent_group=None):
        """Add a task group (e.g., 'healthcare' containing 'medical', 'pharmacy')"""
        self.task_hierarchy[group_name] = {
            "parent": parent_group,
            "tasks": {}
        }

    def add_task(self, group_name, task_name):
        """Add task to a group"""
        if group_name not in self.task_hierarchy:
            self.add_task_group(group_name)

        self.task_hierarchy[group_name]["tasks"][task_name] = {
            "adapter": None,
            "children": []
        }

    def get_active_adapters(self, task_path):
        """Get all adapters in the hierarchy path"""
        adapters = []
        current = task_path

        while current in self.task_hierarchy:
            task_info = self.task_hierarchy[current]
            if task_info["adapter"]:
                adapters.append(task_info["adapter"])
            current = task_info["parent"]

        return adapters

    def forward(self, task_path, *args, **kwargs):
        """Forward with hierarchical adapters"""
        active_adapters = self.get_active_adapters(task_path)

        # Apply adapters in order (parent to child)
        outputs = self.base_model(*args, **kwargs)

        for adapter in reversed(active_adapters):
            outputs = adapter(*args, outputs=outputs, **kwargs)

        return outputs

# Usage
clora = HierarchicalCLoRA(base_model)

# Create hierarchy
clora.add_task_group("healthcare")
clora.add_task_group("medical", "healthcare")
clora.add_task_group("surgery", "medical")

# Add tasks with hierarchy
clora.task_hierarchy["healthcare"]["tasks"]["general"] = {
    "adapter": train_adapter(base_model, healthcare_data)
}

clora.task_hierarchy["medical"]["tasks"]["internal"] = {
    "adapter": train_adapter(base_model, medical_data)
}

clora.task_hierarchy["surgery"]["tasks"]["cardiac"] = {
    "adapter": train_adapter(base_model, surgery_data)
}

# Inference with hierarchy
# Uses: healthcare-general → medical-internal → surgery-cardiac
general_output = clora.forward("healthcare/general", prompt)
medical_output = clora.forward("medical/internal", prompt)
surgery_output = clora.forward("surgery/cardiac", prompt)
```

### 2. Meta-Learning for Task Adaptation

```python
class MetaCLoRA:
    def __init__(self, base_model, meta_learning_rate=0.001):
        self.base_model = base_model
        self.meta_optimizer = torch.optim.Adam(
            base_model.parameters(),
            lr=meta_learning_rate
        )
        self.task_adapters = {}

    def train_on_task(self, task_name, train_data, val_data, steps=100):
        """Train adapter and update base model via meta-learning"""
        # Create new adapter
        adapter = create_lora_adapter(self.base_model)

        # Fast weights adaptation
        fast_params = {n: p.clone() for n, p in self.base_model.named_parameters()}

        for step in range(steps):
            # Sample batch
            batch = sample_batch(train_data)

            # Adapt parameters
            adapted_params = adapt_parameters(
                fast_params,
                adapter,
                batch
            )

            # Compute loss
            loss = compute_loss(adapted_params, batch)

            # Meta-update
            gradients = compute_gradients(loss, fast_params)
            self.meta_optimizer.step(gradients)

        # Save adapted base model
        self.task_adapters[task_name] = {
            "base": copy_of_base_model,
            "adapter": adapter
        }

    def forward(self, task_name, *args, **kwargs):
        """Forward with task-specific meta-learned parameters"""
        if task_name not in self.task_adapters:
            raise ValueError(f"Task {task_name} not trained")

        task_info = self.task_adapters[task_name]

        # Load task-specific base and adapter
        model = task_info["base"]
        adapter = task_info["adapter"]

        return adapter(model, *args, **kwargs)

# Usage
meta_clora = MetaCLoRA(base_model)

# Train on tasks with meta-learning
meta_clora.train_on_task("medical", medical_train, medical_val)
meta_clora.train_on_task("legal", legal_train, legal_val)

# Inference
medical_output = meta_clora.forward("medical", medical_prompt)
legal_output = meta_clora.forward("legal", legal_prompt)
```

### 3. C-LoRA with Memory Compression

```python
class CompressedCLoRA:
    def __init__(self, base_model):
        self.base_model = base_model
        self.compressed_adapters = {}

    def compress_adapter(self, adapter, method="quantization"):
        """Compress LoRA adapter using various methods"""
        if method == "quantization":
            # 8-bit quantization
            return quantize_adapter(adapter, bits=8)
        elif method == "pruning":
            # Sparse adaptation
            return prune_adapter(adapter, sparsity=0.7)
        elif method == "distillation":
            # Distill to smaller adapter
            return distill_adapter(adapter, rank=8)
        else:
            raise ValueError(f"Unknown compression method: {method}")

    def save_compressed(self, adapter, task_name, method="quantization"):
        """Save compressed adapter"""
        compressed = self.compress_adapter(adapter, method)
        self.compressed_adapters[task_name] = compressed
        compressed.save_pretrained(f"{task_name}_{method}")

    def load_compressed(self, task_name, method="quantization"):
        """Load compressed adapter"""
        if task_name not in self.compressed_adapters:
            adapter = load_adapter(f"{task_name}_{method}")
            self.compressed_adapters[task_name] = adapter
        return self.compressed_adapters[task_name]

    def forward(self, task_name, *args, method="quantization", **kwargs):
        """Forward with compressed adapter"""
        adapter = self.load_compressed(task_name, method)
        return adapter(self.base_model, *args, **kwargs)

# Usage
compressed_clora = CompressedCLoRA(base_model)

# Train and compress
medical_adapter = train_adapter(base_model, medical_data)
compressed_clora.save_compressed(medical_adapter, "medical", method="quantization")

# Inference with compressed adapter
output = compressed_clora.forward("medical", prompt, method="quantization")
```

## C-LoRA in Production

### Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 API Gateway                                 │
└─────────────────────────────────────────────────────────────┘
                                │
        ┌─────────────────────────────────────────────────────┐
        │              Task Classifier                         │
        │  - Identifies task from prompt/query                  │
        └─────────────────────────────────────────────────────┘
                                │
        ┌─────────────────────────────────────────────────────┐
        │              Adapter Loader                          │
        │  - Loads compressed task adapters from storage       │
        └─────────────────────────────────────────────────────┘
                                │
        ┌─────────────────────────────────────────────────────┐
        │              Base Model (Frozen)                     │
        │  - Single shared base model                         │
        └─────────────────────────────────────────────────────┘
                                │
        ┌─────────────────────────────────────────────────────┐
        │              Inference Engine                        │
        │  - Applies task-specific adapter                      │
        │  - Generates response                               │
        └─────────────────────────────────────────────────────┘
                                │
        ┌─────────────────────────────────────────────────────┐
        │              Response Cache                          │
        │  - Caches frequent queries                          │
        └─────────────────────────────────────────────────────┘
                                │
                        ┌─────────────────────────────┐
                        │          Client             │
                        └─────────────────────────────┘
```

### Storage Optimization

```python
import zstandard as zstd
import pickle

class CLoRAStorage:
    def __init__(self, storage_path):
        self.storage_path = storage_path
        self.compression = zstd.ZstdCompressor()

    def save_adapter(self, adapter, task_name):
        """Save adapter with compression"""
        data = pickle.dumps(adapter.state_dict())
        compressed = self.compression.compress(data)

        with open(f"{self.storage_path}/{task_name}.clora", "wb") as f:
            f.write(compressed)

    def load_adapter(self, task_name):
        """Load and decompress adapter"""
        with open(f"{self.storage_path}/{task_name}.clora", "rb") as f:
            compressed = f.read()

        data = self.compression.decompress(compressed)
        return pickle.loads(data)

    def get_storage_size(self):
        """Calculate total storage usage"""
        total = 0
        for file in os.listdir(self.storage_path):
            total += os.path.getsize(f"{self.storage_path}/{file}")
        return total / (1024 * 1024)  # MB

# Usage
storage = CLoRAStorage("/path/to/adapters")

# Save adapter
storage.save_adapter(medical_adapter, "medical")

# Load adapter
adapter = storage.load_adapter("medical")

# Check storage
print(f"Total storage: {storage.get_storage_size():.2f} MB")
```

### Continuous Learning Pipeline

```python
import schedule
import time

class ContinuousLearningPipeline:
    def __init__(self, base_model_path, evaluation_data):
        self.base_model_path = base_model_path
        self.evaluation_data = evaluation_data
        self.task_adapters = {}
        self.performance_history = {}

    def evaluate_adapter(self, adapter, task_name):
        """Evaluate adapter performance"""
        results = {}

        for eval_task, eval_data in self.evaluation_data.items():
            if eval_task == task_name:
                # Current task performance
                acc = evaluate(adapter, eval_data)
                results[f"{task_name}_current"] = acc
            else:
                # Backward transfer
                acc = evaluate(adapter, eval_data)
                results[f"{eval_task}_backward"] = acc

        return results

    def check_performance_degradation(self, task_name, threshold=0.05):
        """Check if performance degraded below threshold"""
        current_perf = self.evaluate_adapter(
            self.task_adapters[task_name],
            task_name
        )

        # Compare to historical best
        best_perf = max(self.performance_history.get(task_name, [0]))
        degradation = best_perf - current_perf

        return degradation > threshold

    def retrain_degraded_task(self, task_name, new_data):
        """Retrain task that shows degradation"""
        print(f"Retraining {task_name} due to performance degradation")

        # Load base model
        base_model = FastLanguageModel.from_pretrained(self.base_model_path)

        # Create new adapter
        adapter = FastLanguageModel.get_peft_model(
            base_model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

        # Combine old + new data
        old_data = load_existing_data(task_name)
        combined_data = old_data + new_data

        # Train
        train(adapter, combined_data)

        # Save
        adapter.save_pretrained(f"{task_name}_v2")
        self.task_adapters[task_name] = adapter

        # Update performance history
        perf = self.evaluate_adapter(adapter, task_name)
        self.performance_history[task_name].append(perf)

    def schedule_continuous_learning(self):
        """Schedule regular performance checks and retraining"""
        # Check performance daily
        schedule.every().day.at("03:00").do(self.check_all_tasks)

        # Add new tasks as they arrive
        schedule.every().hour.do(self.check_for_new_tasks)

        # Run forever
        while True:
            schedule.run_pending()
            time.sleep(60)

    def check_all_tasks(self):
        """Check all tasks for performance degradation"""
        for task_name in self.task_adapters:
            if self.check_performance_degradation(task_name):
                # Get new data for this task
                new_data = fetch_new_data(task_name)
                self.retrain_degraded_task(task_name, new_data)

    def check_for_new_tasks(self):
        """Check if new tasks need to be added"""
        new_tasks = detect_new_tasks()

        for task_name in new_tasks:
            print(f"Adding new task: {task_name}")

            # Train initial adapter
            base_model = FastLanguageModel.from_pretrained(self.base_model_path)
            adapter = FastLanguageModel.get_peft_model(
                base_model,
                r=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )

            # Load task data
            task_data = load_task_data(task_name)

            # Train
            train(adapter, task_data)

            # Save
            adapter.save_pretrained(task_name)
            self.task_adapters[task_name] = adapter

            # Initialize performance history
            perf = self.evaluate_adapter(adapter, task_name)
            self.performance_history[task_name] = [perf]

# Usage
pipeline = ContinuousLearningPipeline(
    base_model_path="base_model",
    evaluation_data={
        "medical": medical_val_data,
        "legal": legal_val_data,
        "coding": coding_val_data
    }
)

# Start continuous learning
# pipeline.schedule_continuous_learning()
```

## Resources

- [C-LoRA Paper: Continual Low-Rank Adaptation for Pre-trained Models](https://arxiv.org/abs/2305.02475)
- [C-LoRA arXiv v1: Continual Low-Rank Adaptation for Pre-trained Models](https://arxiv.org/html/2502.17920v1) - Technical details and analysis
- [PEFT Library](https://github.com/huggingface/peft) - Parameter-Efficient Fine-Tuning
- [Unsloth](https://github.com/unslothai/unsloth) - Fast LLM finetuning
- [Continual Learning Benchmarks](https://www.continual-learning.org/)
- [GEM: Gradient Episodic Memory](https://arxiv.org/abs/1703.03910)
- [EWC: Elastic Weight Consolidation](https://arxiv.org/abs/1612.00796)

## Troubleshooting

### Common Issues and Solutions

**Problem**: New task performance is poor
- **Check**: Is the learning rate appropriate?
- **Solution**: Try lower learning rates (1e-4 to 5e-4) for continual learning

**Problem**: Previous tasks performance drops significantly
- **Check**: Is there sufficient regularization?
- **Solution**: Add EWC or GEM regularization, or use rehearsal

**Problem**: Adapter grows too large
- **Check**: Are you using proper compression?
- **Solution**: Use 8-bit quantization or pruning for adapters

**Problem**: Task switching is slow
- **Check**: Are adapters loaded from disk each time?
- **Solution**: Keep frequently used adapters in memory

**Problem**: Memory leaks during continual learning
- **Check**: Are you properly unloading old adapters?
- **Solution**: Use context managers or explicit cleanup

**Problem**: Training time increases with each task
- **Check**: Are you accumulating all training data?
- **Solution**: Use mini-batch sampling from previous tasks

## Future Directions

1. **Automatic Task Discovery**: Identify new tasks from data distribution shifts
2. **Neural Architecture Search for Adapters**: Optimize adapter architecture per task
3. **Federated C-LoRA**: Continual learning across distributed devices
4. **Multimodal C-LoRA**: Extend to vision, audio, and other modalities
5. **Explainable Task Adaptation**: Understand what knowledge is being adapted
6. **Energy-Efficient C-LoRA**: Optimize for edge devices and mobile

## Best Practices

1. **Start with related tasks**: C-LoRA works best when tasks are in the same domain
2. **Monitor backward transfer**: Track performance on all previous tasks
3. **Use regularization**: EWC, GEM, or rehearsal helps prevent forgetting
4. **Compress adapters**: Use quantization or pruning to save storage
5. **Version adapters**: Track adapter versions for rollback capability
6. **Warm start**: Initialize new adapters with knowledge from similar tasks
7. **Validate frequently**: Check performance after each new task
8. **Document tasks**: Keep metadata about what each adapter was trained on
