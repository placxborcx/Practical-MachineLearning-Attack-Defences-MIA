"""
defense_mechanism.py
Four-Layer Hybrid Defence against Membership-Inference Attack.
Author: YoHo
"""

import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import argparse, os, json
import math

# -------------------- Hyper-parameters --------------------
class CFG:
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed         = 42
    epochs       = 50
    batch        = 128
    lr           = 1e-3
    weight_decay = 1e-3          # L2 regularisation (Layer-1)
    dropout_p    = 0.6          # Layer-1 previous 0.5
    temperature  = 6.0           # Layer-2 previous 2 -> 3 -> 4
    noise_std_min = 0.30        # pre 0.2
    noise_std_max = 1.2        # pre 0.6
    clip_max     = 0.6          # Layer-4 previous 0.9 -> 0.85
    round_digit  = 1             # Layer-4 previous 1
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
"""
class DefenceWrapper(nn.Module):
    Wrap a trained model and apply temperature, noise, clip & round.
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
        # We keep logits for downstream softmax; clipping+round 
        return logits
"""

class DefenceWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model.eval()

    @torch.no_grad()
    def forward(self, x):
        """
        Return *sanitised logits* = log(clip-&-round probs).
        This keeps API 介面不變，卻把可被攻擊利用的細節壓掉。
        """
        # 1) 原始 logits + temperature
        raw_logits = self.base(x) / CFG.temperature

        # 2) Adaptive noise（保留你原本的 confidence-based std）
        probs_tmp   = torch.softmax(raw_logits, dim=1)
        max_conf    = probs_tmp.max(dim=1, keepdim=True)[0]
        std         = CFG.noise_std_min + (CFG.noise_std_max-CFG.noise_std_min)*max_conf.pow(2)
        noisy_logits = raw_logits + torch.randn_like(raw_logits) * std

        # 3) 轉機率 → clip → round → 再正規化
        probs = F.softmax(noisy_logits, dim=1)
        probs = torch.clamp(probs, 0.0, CFG.clip_max)
        probs = probs / probs.sum(dim=1, keepdim=True)
        factor = 10 ** CFG.round_digit
        probs = torch.round(probs * factor) / factor
        row_sum   = probs.sum(dim=1)        # shape = [N,1]keepdim=True
        zero_mask = (row_sum == 0)                        # 哪些樣本被 round 成全 0
        if zero_mask.any():
            probs[zero_mask, :] = 1.0 / probs.size(1)        # 換成 uniform

        probs = probs / probs.sum(dim=1, keepdim=True)

        sanitised_logits = torch.log(torch.clamp(probs, min=1e-6))
        return sanitised_logits


    @torch.no_grad()
    def defended_probs(self, x):
        probs  = F.softmax(self.forward(x), dim=1)    
      
        # Layer-4: 簡化的 clip & round
        # Step 1: 直接 clamp 到上限
        probs = torch.clamp(probs, min=0.0, max=CFG.clip_max)
        
        # Step 2: 重新正規化
        probs = probs / probs.sum(dim=1, keepdim=True)
        
        # Step 3: 四捨五入
        factor = 10 ** CFG.round_digit
        probs = torch.round(probs * factor) / factor
        
        # Step 4: 最終正規化以確保機率和為 1
        probs = probs / probs.sum(dim=1, keepdim=True)
        
        # Step 5: 添加均勻噪音來進一步平滑分佈
        uniform_noise = torch.rand_like(probs) * 0.01  # 小量均勻噪音
        probs = probs + uniform_noise
        probs = probs / probs.sum(dim=1, keepdim=True)
        
        return probs
    
    """
    @torch.no_grad()
    def defended_probs(self, x):
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        
        # Layer-4: clip & round
        # Step 1: Handle clipping more robustly
        for i in range(probs.size(0)):
            max_prob, max_idx = probs[i].max(dim=0)
            
            if max_prob > CFG.clip_max:
                # Calculate how much we need to redistribute
                excess = max_prob - CFG.clip_max
                probs[i, max_idx] = CFG.clip_max
                
                # Redistribute excess to other classes proportionally
                # Get mask for non-max indices
                mask = torch.ones_like(probs[i], dtype=torch.bool)
                mask[max_idx] = False
                
                # Only redistribute to non-zero probabilities
                non_max_probs = probs[i, mask]
                if non_max_probs.sum() > 0:
                    # Add excess proportionally to other classes
                    probs[i, mask] = non_max_probs + (excess * non_max_probs / non_max_probs.sum())
                else:
                    # Edge case: if all other probs are 0, distribute evenly
                    num_other_classes = probs.size(1) - 1
                    if num_other_classes > 0:
                        probs[i, mask] = excess / num_other_classes
        
        # Step 2: Round probabilities
        factor = 10 ** CFG.round_digit
        probs = torch.round(probs * factor) / factor
        
        # Step 3: Ensure no probability exceeds clip_max after rounding
        probs = torch.clamp(probs, min=0.0, max=CFG.clip_max)
        
        # Step 4: Final renormalization
        probs = probs / probs.sum(dim=1, keepdim=True)
        
        # Step 5: Final check and adjustment if needed
        max_probs_final = probs.max(dim=1)[0]
        if (max_probs_final > CFG.clip_max).any():
            # If still exceeding, force clamp and renormalize
            probs = torch.clamp(probs, min=0.0, max=CFG.clip_max)
            probs = probs / probs.sum(dim=1, keepdim=True)
        
        return probs
    """

# ------------------- Training routine --------------------
def train_defended_target():
    torch.set_num_threads(os.cpu_count())     # run all 4 core cpu
    torch.backends.mkldnn.enabled = True        #enhance speed
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
    parser.add_argument("--epochs",  type=int, default=CFG.epochs,
                        help="override number of epochs")
    parser.add_argument("--batch",   type=int, default=CFG.batch,
                        help="override batch size")
    args = parser.parse_args()

    CFG.epochs = args.epochs
    CFG.batch  = args.batch

    if args.train:
        train_defended_target()
