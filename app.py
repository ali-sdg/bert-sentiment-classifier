import gradio as gr
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Configuratio
MODEL_ID = "ali-sdg/bert-sentiment-classifier" 
LABELS   = ["World", "Sports", "Business", "Sci/Tech"]
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load Model
print(f"Loading model from {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model     = model.to(DEVICE)
model.eval()
print(f"Model loaded on {DEVICE}")

# Inference
def classify_news(text: str) -> dict:
    """
    Classifies news text into one of the predefined categories.
    
    Args:‍
        text (str): The news headline or article text.
        
    Returns:
        dict: A dictionary mapping category labels to their confidence scores.
    """
    if not text or not text.strip():
        return {label: 0.0 for label in LABELS}

    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Inference
    with torch.no_grad():
        logits = model(**inputs).logits

    # Calculate probabilities
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    return {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}

# Gradio UI
examples = [
    ["Apple announces M4 chip with breakthrough AI performance"],
    ["Real Madrid wins Champions League final against Manchester City"],
    ["Federal Reserve raises interest rates amid inflation concerns"],
    ["UN Security Council meets to discuss Middle East crisis"],
    ["Scientists discover new method to reverse aging in mice"],
    ["Tesla reports record quarterly earnings, stock surges 10%"],
]

with gr.Blocks(title="News Classifier", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # News Category Classifier
    **DistilBERT fine-tuned on AG News** - Classifies into 4 categories
    """)

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="News Headline or Article",
                placeholder="e.g. NASA launches James Webb telescope upgrade...",
                lines=4,
            )
            with gr.Row():
                submit_btn = gr.Button("Classify", variant="primary")
                clear_btn  = gr.Button("Clear")

        with gr.Column():
            label_output = gr.Label(
                label="Category Prediction",
                num_top_classes=4,
            )

    gr.Examples(
        examples=examples,
        inputs=text_input,
        label="Try these examples",
    )

    gr.Markdown("""
    ---
    **Model:** `distilbert-base-uncased` fine-tuned on [AG News](https://huggingface.co/datasets/ag_news)  
    **Accuracy:** ~94% on test set | **Categories:** World | Sports | Business | Sci/Tech
    """)

    # Event handlers
    submit_btn.click(fn=classify_news, inputs=text_input, outputs=label_output)
    text_input.submit(fn=classify_news, inputs=text_input, outputs=label_output)
    clear_btn.click(fn=lambda: ("", None), outputs=[text_input, label_output])


if __name__ == "__main__":
    demo.launch()
