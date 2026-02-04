import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 学習済みモデルのパス
model_path = "ch09/out/87/checkpoint-3159"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

test_sentences = [
    "The movie was full of incomprehensibilities.",
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish."
]

for text in test_sentences:
    inputs = tokenizer(test_sentences, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)

    preds = probs.argmax(dim=-1)

    for text, pred, prob in zip(test_sentences, preds, probs):
        label = "Positive" if pred.item() == 1 else "Negative"
        print(f"{text}\nPrediction: {label} ({prob[pred].item():.2%})\n")
