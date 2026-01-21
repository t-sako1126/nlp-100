import torch
import torch.nn as nn
from p70 import vocab_size, emb_dim, emb
from p72 import Bowmodel

#モデルの学習
model = Bowmodel(vocab_size, emb_dim)
model.embedding.weight.data = emb  # 埋め込み層に事前学習済みベクトルをセット 
criterion = nn.BCELoss()  # 二値分類の損失関数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 最適化手法  
BATCH_SIZE = 32
EPOCHS = 5
def train_model(train_data):
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for i in range(0, len(train_data), BATCH_SIZE):
            batch = train_data[i:i+BATCH_SIZE]
            input_ids = nn.utils.rnn.pad_sequence(
                [item["input_ids"] for item in batch],
                batch_first=True,
                padding_value=0
            )
            labels = torch.cat([item["label"] for item in batch], dim=0)

            optimizer.zero_grad()
            outputs = model(input_ids).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch)

        avg_loss = total_loss / len(train_data)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")