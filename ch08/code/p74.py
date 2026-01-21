import torch
from torch.utils.data import DataLoader
from p70 import emb, vocab_size, emb_dim
from p72 import Bowmodel
from p71 import dev_data  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_one(batch):
    item = batch[0]
    input_ids = item["input_ids"].unsqueeze(0)
    label = item["label"].view(1, 1)          
    return input_ids, label

def main():
    model = Bowmodel(vocab_size, emb_dim).to(device)
    model.embedding.weight.data = emb.to(device)
    model.embedding.weight.requires_grad = False
    model.load_state_dict(torch.load("ch08/ch08_model.pth", map_location=device))
    model.eval()

    dev_dl = DataLoader(dev_data, batch_size=1, shuffle=False, collate_fn=collate_one)

    correct = 0
    total = 0

    with torch.no_grad():
        for input_ids, label in dev_dl:
            input_ids = input_ids.to(device)
            label = label.to(device)                 

            prob = model(input_ids)                  
            pred = (prob >= 0.5).float()             

            correct += (pred == label).sum().item()
            total += label.numel()

    acc = correct / total
    print(f"dev accuracy:", acc)

if __name__ == "__main__":
    main()
