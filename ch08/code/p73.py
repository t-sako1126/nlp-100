import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from p70 import emb, vocab_size, emb_dim          # E: (|V|+1, 300) torch.Tensor
from p71 import train_data                      # [{'input_ids':..., 'label':...}, ...]
from p72 import Bowmodel                        # embedding→平均→linear→sigmoid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_one(batch):
    item = batch[0]
    input_ids = item["input_ids"].unsqueeze(0)  # (1, L)
    label = item["label"].view(1, 1)            # (1, 1)
    return input_ids, label



def main():
    model = Bowmodel(vocab_size, emb_dim)

    # 事前学習済み埋め込みをセット＆固定（73の条件）
    model.embedding.weight.data = emb.to(device)
    model.embedding.weight.requires_grad = False

    model = model.to(device)

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.linear.parameters(), lr=0.1)

    train_dl = DataLoader(train_data, batch_size=1, shuffle=True, collate_fn=collate_one)

    EPOCHS = 10
    patience = 2
    best_loss = float("inf")
    bad_epochs = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for input_ids, label in train_dl:
            input_ids = input_ids.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            prob = model(input_ids)        # (1,1)
            loss = loss_fn(prob, label)    # labelも(1,1)なので一致
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dl)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}", flush=True)

        # 早期終了: 損失が改善しないepochが続いたら打ち切り
        if avg_loss < best_loss:
            best_loss = avg_loss
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(
                    f"Early stopping at epoch {epoch+1} (best_loss={best_loss:.4f})",
                    flush=True,
                )
                break

    torch.save(model.state_dict(), "ch08/ch08_model.pth")

if __name__ == "__main__":
    main()
