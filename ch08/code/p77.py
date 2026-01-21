import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from p70 import emb, vocab_size, emb_dim
from p71 import train_data, dev_data
from p72 import Bowmodel

# ===== device =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== collate (75と同じ) =====
def collate(batch):
    batch = sorted(batch, key=lambda x: x["input_ids"].numel(), reverse=True)

    lengths = [ex["input_ids"].numel() for ex in batch]
    max_len = lengths[0]
    B = len(batch)

    input_ids = torch.zeros((B, max_len), dtype=torch.long)
    labels = torch.stack([ex["label"] for ex in batch]).view(B, 1).float()

    for i, ex in enumerate(batch):
        l = ex["input_ids"].numel()
        input_ids[i, :l] = ex["input_ids"]

    return {"input_ids": input_ids, "label": labels}

# ===== dev 評価 =====
@torch.no_grad()
def evaluate(model, dev_dl):
    model.eval()
    correct, total = 0, 0
    for batch in dev_dl:
        x = batch["input_ids"].to(device)
        y = batch["label"].to(device)

        prob = model(x)
        pred = (prob >= 0.5).float()

        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total

def main():
    print("device:", device)

    # ===== model =====
    model = Bowmodel(vocab_size, emb_dim).to(device)

    # 事前学習済み埋め込みをGPUに載せて固定
    model.embedding.weight.data = emb.to(device)
    model.embedding.weight.requires_grad = False

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.linear.parameters(), lr=0.1)

    # ===== DataLoader =====
    batch_size = 32
    train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate)
    dev_dl   = DataLoader(dev_data,   batch_size=batch_size, shuffle=False, collate_fn=collate)

    # ===== training =====
    EPOCHS = 10
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in train_dl:
            x = batch["input_ids"].to(device)
            y = batch["label"].to(device)

            optimizer.zero_grad()
            prob = model(x)
            loss = loss_fn(prob, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dl)
        dev_acc = evaluate(model, dev_dl)

        print(f"Epoch {epoch+1}/{EPOCHS}  Loss: {avg_loss:.4f}  DevAcc: {dev_acc:.4f}")

    torch.save(model.state_dict(), "ch08/ch08_model_gpu.pth")

if __name__ == "__main__":
    main()
