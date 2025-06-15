"""
defense_mechanism.py
Four-Layer Hybrid Defence against Membership-Inference Attack.
Author: YoHo
"""

import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import argparse, os, json

# -------------------- Hyper-parameters --------------------
class CFG:
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed         = 42
    epochs       = 50
    batch        = 128
    lr           = 1e-3
    weight_decay = 1e-3          # L2 regularisation (Layer-1)
    dropout_p    = 0.5           # Layer-1
    temperature  = 2.0           # Layer-2
    noise_std    = 0.25          # Layer-3
    clip_max     = 0.90          # Layer-4
    round_digit  = 2             # Layer-4
    save_path    = "./defended_target.pth"
    json_out     = "./defense_config.json"

# Set seed
torch.manual_seed(CFG.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(CFG.seed)

# -------------------- Model with Dropout -----------------
class DefendedCNN(nn.Module):
    """Simple CNN + dropout for regularisation."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(CFG.dropout_p),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*14*14, 256), nn.ReLU(),
            nn.Dropout(CFG.dropout_p),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ------------- Defence wrapper (Layers 2-4) --------------
class DefenceWrapper(nn.Module):
    """Wrap a trained model and apply temperature, noise, clip & round."""
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model.eval()  # inference mode

    @torch.no_grad()
    def forward(self, x):
        logits = self.base(x)
        # Layer-2: temperature
        logits = logits / CFG.temperature
        # Layer-3: Gaussian noise
        if CFG.noise_std > 0:
            logits += torch.randn_like(logits) * CFG.noise_std
        # We keep logits for downstream softmax; clipping+round 用機率實作
        return logits

    @torch.no_grad()
    def defended_probs(self, x):
        logits = self.forward(x)
        probs  = F.softmax(logits, dim=1)
        # Layer-4: clip & round
        probs  = torch.clamp(probs, 1e-7, CFG.clip_max)
        probs  = probs / probs.sum(dim=1, keepdim=True)
        factor = 10 ** CFG.round_digit
        probs  = torch.round(probs * factor) / factor
        probs  = probs / probs.sum(dim=1, keepdim=True)
        return probs

# ------------------- Training routine --------------------
def train_defended_target():
    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
    train_set = datasets.MNIST("./data", train=True, download=True, transform=trans)
    train_len = int(0.9 * len(train_set))
    val_len   = len(train_set) - train_len
    train_ds, val_ds = random_split(train_set, [train_len, val_len])

    train_ld = DataLoader(train_ds, batch_size=CFG.batch, shuffle=True)
    val_ld   = DataLoader(val_ds, batch_size=CFG.batch)

    model = DefendedCNN().to(CFG.device)
    opt   = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    best_val = 0

    for ep in range(CFG.epochs):
        model.train()
        for x, y in train_ld:
            x, y = x.to(CFG.device), y.to(CFG.device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()

        # --- validation ---
        model.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for x, y in val_ld:
                x, y = x.to(CFG.device), y.to(CFG.device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total   += y.size(0)
        acc = correct / total
        if acc > best_val:
            best_val = acc
            torch.save(model.state_dict(), CFG.save_path)
        if ep % 10 == 0:
            print(f"[Ep{ep:02d}] val_acc={acc:.4f}")

    # Save defence config for reproducibility
    json.dump(vars(CFG), open(CFG.json_out, "w"), indent=2)
    print(f"✔ Done. Best val_acc={best_val:.4f}, model → {CFG.save_path}")

# -------------------- Entry-point ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="train defended target")
    args = parser.parse_args()
    if args.train:
        train_defended_target()
