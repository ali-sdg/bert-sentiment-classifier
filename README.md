
A fine-tuned DistilBERT model for multi-class news classification, deployed as an interactive web app on Hugging Face Spaces.

[Live Demo](https://huggingface.co/spaces/ali-sdg/bert-sentiment-classifier) | [Model on HuggingFace](https://huggingface.co/ali-sdg/bert-sentiment-classifier) 

---

## Overview

This project fine-tunes `distilbert-base-uncased` on the AG News dataset to classify news articles into 4 categories: World, Sports, Business, and Sci/Tech.

The goal was both educational and practical — understanding the full pipeline from raw text to a deployed NLP model, including tokenization, fine-tuning with the Trainer API, and deployment on Hugging Face Spaces with a Gradio interface.

---

## Project Structure

```
.
├── src/
│   └── train.ipynb        # Fine-tuning notebook (data loading, training, evaluation)
├── app.py                 # Gradio web interface
├── requirements.txt       # Dependencies
├── LICENSE
└── README.md
```

The trained model weights are not stored in this repository. They are hosted on Hugging Face Hub at `ali-sdg/bert-sentiment-classifier`.

---

## Model

| Property | Value |
|---|---|
| Base model | distilbert-base-uncased |
| Task | Text classification (4 classes) |
| Dataset | AG News |
| Training samples | 20,000 |
| Test accuracy | ~94% |
| F1 score (weighted) | ~94% |

### Architecture

The base DistilBERT encoder processes the input and the `[CLS]` token representation is passed to a classification head:

```
Input Text
    -> Tokenizer (input_ids + attention_mask)
    -> DistilBERT Encoder (768-dim CLS output)
    -> Linear(768, 768) -> ReLU -> Dropout
    -> Linear(768, 4)
    -> Softmax
    -> Predicted class
```

---

## Training Details

| Parameter | Value |
|---|---|
| Learning rate | 2e-5 |
| Batch size | 32 |
| Epochs | 3 |
| Max sequence length | 128 |
| Warmup ratio | 0.1 |
| Weight decay | 0.01 |
| Precision | FP16 |
| Optimizer | AdamW |

Training was done on Google Colab using the HuggingFace Trainer API with dynamic padding via `DataCollatorWithPadding`.

---

## Usage

### Run the app locally

```bash
git clone https://github.com/ali-sdg/news-category-classifier
cd news-category-classifier
pip install -r requirements.txt
python app.py
```

### Use the model directly

```python
from transformers import pipeline

pipe = pipeline(
    "text-classification",
    model="ali-sdg/bert-sentiment-classifier"
)

pipe("NASA announces new Mars mission with AI-powered rover")
# [{'label': 'Sci/Tech', 'score': 0.95}]
```

### Manual inference

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_id = "ali-sdg/bert-sentiment-classifier"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
model.eval()

text = "Federal Reserve raises interest rates amid inflation concerns"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

with torch.no_grad():
    logits = model(**inputs).logits

probs = torch.softmax(logits, dim=-1)
label = model.config.id2label[probs.argmax().item()]
confidence = probs.max().item()

print(f"{label} ({confidence:.2%})")
# Business (96.12%)
```

---

## Tech Stack

- PyTorch
- HuggingFace Transformers
- HuggingFace Datasets
- HuggingFace Trainer API
- Gradio
- Hugging Face Spaces

---

