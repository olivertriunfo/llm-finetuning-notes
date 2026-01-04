# Training Datasets for LLM Finetuning (2026)

High-quality datasets are crucial for effective LLM finetuning. This document covers datasets suitable for various finetuning tasks in 2026.

## Dataset Selection Criteria

When choosing a dataset for finetuning, consider:

1. **Relevance**: Does the dataset match your target use case?
2. **Quality**: Is the data well-curated and accurate?
3. **Size**: Is it large enough for your model size?
4. **License**: Are you allowed to use it commercially?
5. **Diversity**: Does it cover a broad range of topics?
6. **Bias**: Does it contain harmful biases or stereotypes?

## Popular Datasets for LLM Finetuning

### General Purpose Datasets

#### 1. RefinedWeb
- **Description**: Filtered and deduplicated web crawl data
- **Size**: 3+ TB of text (2026 edition)
- **Use Case**: General LLM finetuning, instruction following
- **Quality**: High-quality after extensive filtering
- **License**: CC BY-NC 4.0
- **Access**: [Hugging Face](https://huggingface.co/datasets/tiiuae/falcon-refinedweb)

#### 2. Pile
- **Description**: Diverse collection of 22 subsets from various sources
- **Size**: 825 GB compressed
- **Use Case**: General knowledge, research
- **Subsets**: Books, GitHub, Wikipedia, arXiv, etc.
- **License**: Mix of licenses, check each subset
- **Access**: [Hugging Face](https://huggingface.co/datasets/EleutherAI/the_pile)

#### 3. C4 (Colossal Clean Crawled Corpus)
- **Description**: Web crawl data filtered for high-quality English text
- **Size**: 750 GB
- **Use Case**: Pretraining, general finetuning
- **Quality**: Cleaned and deduplicated
- **License**: CC BY 3.0
- **Access**: [TFDS](https://www.tensorflow.org/datasets/catalog/c4)

### Instruction Tuning Datasets

#### 1. Alpaca
- **Description**: 52K instruction-following examples generated with text-davinci-003
- **Size**: ~52K examples
- **Use Case**: Instruction finetuning, chatbot training
- **Format**: Input-output pairs
- **License**: CC BY-NC-SA 4.0
- **Access**: [Hugging Face](https://huggingface.co/datasets/tloen/alpaca)

#### 2. Guanaco
- **Description**: Improved version of Alpaca with better quality examples
- **Size**: 60K+ examples
- **Use Case**: Instruction finetuning
- **Quality**: Filtered and improved
- **License**: CC BY-NC-SA 4.0
- **Access**: [Hugging Face](https://huggingface.co/datasets/TheBloke/guanaco-65B-GPTQ)

#### 3. Self-Instruct
- **Description**: 82K examples generated through self-instruction
- **Size**: 82K examples
- **Use Case**: Instruction finetuning, conversational agents
- **Format**: Multi-turn conversations
- **License**: MIT
- **Access**: [Hugging Face](https://huggingface.co/datasets/yizhongw/self-instruct)

#### 4. ShareGPT
- **Description**: 50K+ ChatGPT conversations from ShareGPT
- **Size**: ~50K conversations
- **Use Case**: Chatbot training, conversational AI
- **Format**: Multi-turn dialogues
- **License**: CC BY-NC-SA 4.0
- **Access**: [Hugging Face](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)

#### 5. OpenAssistant (OASST1 & OASST2)
- **Description**: Assistant-style conversation dataset
- **Size**: 16K+ conversations (OASST1), 50K+ (OASST2)
- **Use Case**: Chatbot training, assistant models
- **Format**: Multi-turn, human-human conversations
- **License**: Apache 2.0
- **Access**: [Hugging Face](https://huggingface.co/datasets/OpenAssistant/oasst1)

### Domain-Specific Datasets

#### Medical
- **MediquAD**: Medical question answering
  - **Access**: [Hugging Face](https://huggingface.co/datasets/healthsearch/mediquad)

- **PubMedQA**: Biomedical question answering
  - **Access**: [Hugging Face](https://huggingface.co/datasets/pubmed_qa)

#### Legal
- **LexGLUE**: Legal benchmarks
  - **Access**: [Hugging Face](https://huggingface.co/datasets/lex_glue)

- **CaseHold**: Legal reasoning
  - **Access**: [Hugging Face](https://huggingface.co/datasets/casehold)

#### Coding
- **CodeParrot**: Code completion dataset
  - **Access**: [Hugging Face](https://huggingface.co/datasets/codeparrot/codeparrot)

- **StarcoderData**: StarCoder training data
  - **Access**: [Hugging Face](https://huggingface.co/datasets/bigcode/starcoderdata)

#### Mathematics
- **MathQA**: Math word problems
  - **Access**: [Hugging Face](https://huggingface.co/datasets/math_qa)

- **GSM8K**: Grade school math problems
  - **Access**: [Hugging Face](https://huggingface.co/datasets/gsm8k)

#### Finance
- **FinBERT**: Financial text
  - **Access**: [Hugging Face](https://huggingface.co/datasets/yanolau/finbert)

- **TREC-Finance**: Financial question answering
  - **Access**: [Hugging Face](https://huggingface.co/datasets/trec-finance)

### Synthetic Datasets

#### 1. Evol-Instruct
- **Description**: Evolved instruction dataset using reinforcement learning
- **Size**: 21K+ examples
- **Use Case**: Advanced instruction finetuning
- **Quality**: Generated with evolutionary algorithms
- **License**: Apache 2.0
- **Access**: [Hugging Face](https://huggingface.co/datasets/teknium/Evol-Instruct-70k)

#### 2. LongForm
- **Description**: Long-form question answering dataset
- **Size**: 24K+ examples
- **Use Case**: Long context finetuning
- **Format**: Questions with detailed answers
- **License**: CC BY-NC-SA 4.0
- **Access**: [Hugging Face](https://huggingface.co/datasets/akoksal/LongForm)

#### 3. GPTeacher
- **Description**: Dataset generated by GPT-4 as teacher
- **Size**: 60K+ examples
- **Use Case**: High-quality instruction finetuning
- **Quality**: Generated with GPT-4
- **License**: MIT
- **Access**: [Hugging Face](https://huggingface.co/datasets/THUDM/GPTeacher)

## Creating Custom Datasets

### Data Collection

1. **Web Scraping** (with respect to robots.txt and terms of service)
   ```python
   import requests
   from bs4 import BeautifulSoup

   def scrape_website(url):
       response = requests.get(url)
       soup = BeautifulSoup(response.text, 'html.parser')
       return soup.get_text()
   ```

2. **APIs** (Twitter, Reddit, StackExchange, etc.)
   ```python
   import requests

   def fetch_tweets(api_key, query, count=100):
       url = f"https://api.twitter.com/2/tweets/search/recent?query={query}&max_results={count}"
       headers = {"Authorization": f"Bearer {api_key}"}
       response = requests.get(url, headers=headers)
       return response.json()
   ```

3. **Human Annotation** (for high-quality custom datasets)

### Data Processing

```python
from datasets import Dataset, DatasetDict
import pandas as pd

# Load from CSV
df = pd.read_csv("custom_data.csv")

# Create Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Split into train/test
dataset = dataset.train_test_split(test_size=0.1)

# Save
dataset.save_to_disk("custom_dataset")
```

### Data Quality Checks

```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("my_dataset")

# Check for duplicates
dataset = dataset.remove_duplicates()

# Filter based on length
dataset = dataset.filter(lambda x: len(x["text"]) > 100)

# Check language (using langdetect)
from langdetect import detect

def is_english(text):
    try:
        return detect(text) == "en"
    except:
        return False

dataset = dataset.filter(lambda x: is_english(x["text"]))
```

### Data Formatting for Instruction Tuning

```python
# Convert plain text to instruction format
def format_instruction(text, instruction="Summarize the following text:"):
    return f"### Instruction:\n{instruction}\n\n### Input:\n{text}\n\n### Response:\n"

# Apply to dataset
formatted_dataset = dataset.map(lambda x: {
    "instruction": format_instruction(x["text"])
})
```

## Dataset Evaluation

### Quality Metrics

1. **Diversity**: Measure vocabulary richness
   ```python
   from collections import Counter

   def calculate_vocab_richness(texts):
       words = " ".join(texts).split()
       unique_words = len(set(words))
       total_words = len(words)
       return unique_words / total_words
   ```

2. **Relevance**: Human evaluation or keyword matching

3. **Bias Detection**: Check for underrepresented groups
   ```python
   from bias_detection import BiasDetector

   detector = BiasDetector()
   biases = detector.analyze(dataset)
   ```

4. **Toxicity**: Use Perspective API or Detoxify
   ```python
   from detoxify import Detoxify

   model = Detoxify("original")
   results = model.predict(dataset["text"])
   ```

### Dataset Statistics

```python
# Calculate statistics
def print_stats(dataset):
    print(f"Total examples: {len(dataset)}")
    print(f"Avg length: {sum(len(x['text']) for x in dataset) / len(dataset):.2f}")
    print(f"Min length: {min(len(x['text']) for x in dataset)}")
    print(f"Max length: {max(len(x['text']) for x in dataset)}")

print_stats(dataset)
```

## Data Augmentation

### Back Translation

```python
from transformers import pipeline

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")

# English -> French -> English
def back_translate(text):
    fr_text = translator(text, max_length=512)[0]['translation_text']
    en_text = translator(fr_text, max_length=512, src_lang="fr")[0]['translation_text']
    return en_text

# Apply to dataset
dataset = dataset.map(lambda x: {"augmented_text": back_translate(x["text"])})
```

### Synonym Replacement

```python
from nltk.corpus import wordnet
import random

def synonym_replacement(text, n=3):
    words = text.split()
    for _ in range(n):
        if len(words) == 0:
            break
        idx = random.randint(0, len(words) - 1)
        word = words[idx]

        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace("_", " ").lower()
                if synonym != word and synonym.isalpha():
                    synonyms.append(synonym)

        if synonyms:
            words[idx] = random.choice(synonyms)

    return " ".join(words)
```

### Paraphrasing

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

paraphraser = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws")

def paraphrase(text):
    return paraphraser(text, max_length=512, num_beams=5, num_return_sequences=1)[0]['generated_text']

# Apply to dataset
dataset = dataset.map(lambda x: {"paraphrased": paraphrase(x["text"])})
```

## Best Practices for Dataset Usage

1. **Mix Multiple Datasets**: Combine general and domain-specific datasets
   ```python
   from datasets import concatenate_datasets

   general = load_dataset("general_dataset")
   domain = load_dataset("domain_dataset")

   combined = concatenate_datasets([general, domain])
   ```

2. **Stratified Sampling**: Ensure balanced representation
   ```python
   from datasets import DatasetDict

   # Split by category
   train = DatasetDict()
   for category in dataset.unique("category"):
       cat_data = dataset.filter(lambda x: x["category"] == category)
       train[category] = cat_data.shuffle().select(range(1000))
   ```

3. **Continuous Evaluation**: Monitor performance on held-out sets
   ```python
   import evaluate

   metric = evaluate.load("accuracy")

   def compute_metrics(eval_pred):
       predictions, labels = eval_pred
       return metric.compute(predictions=predictions, references=labels)
   ```

4. **Privacy Considerations**: Anonymize sensitive data
   ```python
   import re

   def anonymize(text):
       # Remove names
       text = re.sub(r"\b[A-Z][a-z]+\s[A-Z][a-z]+\b", "[NAME]", text)
       # Remove emails
       text = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[EMAIL]", text)
       return text

   dataset = dataset.map(lambda x: {"anonymized": anonymize(x["text"])})
   ```

5. **Version Control**: Track dataset versions
   ```bash
   # Save with version
   dataset.save_to_disk("dataset_v1.0")

   # Load specific version
   dataset = Dataset.load_from_disk("dataset_v1.0")
   ```

## Resources

- [Hugging Face Datasets Hub](https://huggingface.co/datasets)
- [TFDS (TensorFlow Datasets)](https://www.tensorflow.org/datasets)
- [Parquet Format](https://parquet.apache.org/) - Efficient columnar storage
- [Dataset Card Guidelines](https://huggingface.co/docs/datasets/dataset_card)
- [Deduplication Tools](https://github.com/facebookresearch/Longformer/tree/main/deduplication)

## 2026 Dataset Trends

1. **Longer Context**: Datasets with 8K+ token sequences
2. **Multimodal**: Text-image pairs for vision-language models
3. **Synthetic Data**: GPT-4 generated instruction datasets
4. **Domain Specialization**: High-quality vertical-specific datasets
5. **Efficiency**: Compressed and filtered datasets for faster training

## License Considerations

Always check dataset licenses before use:

- **CC BY**: Allow commercial use with attribution
- **CC BY-SA**: Allow commercial use with attribution and share-alike
- **CC BY-NC**: Non-commercial only
- **MIT/Apache**: Generally permissive
- **Proprietary**: Requires license

For production use, consider:
- Creating your own datasets
- Using datasets with commercial-friendly licenses
- Getting explicit permission for sensitive applications
