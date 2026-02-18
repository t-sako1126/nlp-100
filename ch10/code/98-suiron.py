import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# =========================
# 設定
# =========================
MODEL_PATH = "ch10/out/qwen-sst2"
DEV_PATH = "ch07/SST-2/dev.tsv"

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# モデル読み込み
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto"
)

model.eval()

# =========================
# データ読み込み
# =========================
df = pd.read_csv(DEV_PATH, sep="\t", header=0)

# =========================
# 推論関数
# =========================
def build_prompt(sentence):
    prompt = f"""Decide if the sentiment is positive or negative.
Answer with only one word: positive or negative.

Sentence: {sentence}
Answer:"""

    messages = [
        {"role": "user", "content": prompt}
    ]

    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return chat_text


@torch.no_grad()
def predict(sentence):

    chat_text = build_prompt(sentence)

    inputs = tokenizer(
        chat_text,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=5,
        do_sample=False,
    )

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    ).strip().lower()

    # positive / negative を抽出
    if "positive" in generated:
        return 1
    elif "negative" in generated:
        return 0
    else:
        return -1


# =========================
# 評価
# =========================
correct = 0
total = 0

for _, row in tqdm(df.iterrows(), total=len(df)):
    pred = predict(row["sentence"])
    gold = int(row["label"])

    if pred == gold:
        correct += 1

    total += 1

acc = correct / total

print(f"Accuracy: {acc:.4f}")
