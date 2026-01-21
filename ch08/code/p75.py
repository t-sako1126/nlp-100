import torch
from p71 import train_data

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

def main():
    batch = train_data[:4]
    out = collate(batch)

    print("input_ids:")
    print(out["input_ids"])
    print("\nlabel:")
    print(out["label"])

if __name__ == "__main__":
    main()
