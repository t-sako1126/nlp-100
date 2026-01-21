import torch
import torch.nn as nn

class Bowmodel(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.linear = nn.Linear(emb_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids):
        """
        input_ids: (L,) or (B, L)
        """
        x = self.embedding(input_ids)   # (L, D) or (B, L, D)
        mean = x.mean(dim=-2)           # 単語方向で平均
        prob = self.sigmoid(self.linear(mean))
        return prob
       
       
if __name__ == "__main__":
 from p70 import vocab_size, emb_dim, emb
 from torchinfo import summary
 model = Bowmodel(vocab_size, emb_dim)
 model.embedding.weight.data = emb  # 埋め込み層に事前学習済みベクトルをセット
 
 BATCH = 32
 LEN = 128
 summary(
    model,
    input_size=(BATCH, LEN),   # input_ids の形
    dtypes=[torch.long],           # Embeddingは long のIDを受け取る
    col_names=("output_size", "num_params"),
    verbose=2
)