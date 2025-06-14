"""
Purpose: Minimal PyTorch implementation of a black-box Membership Inference Attack.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split, ConcatDataset

from torchvision import datasets, transforms
from torchvision.transforms import functional as TF

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

# -----------------------------
# 1.  Global config
# -----------------------------
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SHADOW    = 128
BATCH_ATTACK    = 256
EPOCHS_SHADOW   = 3           # keep small for demo; raise for real study
EPOCHS_ATTACK   = 10
N_SHADOW        = 4
NUM_CLASSES     = 10
SEED            = 1337
torch.manual_seed(SEED)
np.random.seed(SEED)

# -----------------------------
# 2.  Simple CNN (matches "common CNN patterns")
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1), nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*12*12, 128), nn.ReLU(),
            nn.Linear(128, num_classes))
    def forward(self, x):
        return self.fc(self.conv(x))

# -----------------------------
# 3.  Helper – entropy & loss
# -----------------------------
def softmax_entropy(logits):
    p = F.softmax(logits, dim=1)
    return -(p * torch.log(p + 1e-12)).sum(dim=1)

def cross_entropy_loss(logits, y):
    return F.cross_entropy(logits, y, reduction="none")

# -----------------------------
# 4.  Load datasets & build auxiliary pool
# -----------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # ensures CIFAR → 1-ch
    transforms.Resize(28),
    transforms.ToTensor()])

mnist_train = datasets.MNIST(root="./data", train=True, download=True,
                             transform=transform)
fashion    = datasets.FashionMNIST(root="./data", train=True, download=True,
                                   transform=transform)
kmnist     = datasets.KMNIST(root="./data", train=True, download=True,
                             transform=transform)
cifar10    = datasets.CIFAR10(root="./data", train=True, download=True,
                              transform=transform)

# Synthetic Gaussian noise as non-member
def gaussian_noise_dataset(n, shape=(1, 28, 28)):
    data = torch.randn(n, *shape)
    data = torch.clamp((data - data.min()) / (data.max()-data.min()+1e-6), 0, 1)
    return TensorDataset(data, torch.zeros(n, dtype=torch.long))

noise_ds = gaussian_noise_dataset(2000)

# Split 60 % MNIST into shadow pools
mnist_indices = np.random.permutation(len(mnist_train))[: int(0.60 * len(mnist_train))]
mnist_aux     = Subset(mnist_train, mnist_indices)

nonmember_ds = ConcatDataset([
    Subset(fashion,   np.random.choice(len(fashion), 8000,  replace=False)),
    Subset(kmnist,    np.random.choice(len(kmnist),  4000,  replace=False)),
    Subset(cifar10,   np.random.choice(len(cifar10), 4000,  replace=False)),
    noise_ds])

# -----------------------------
# 5.  Train shadow models
# -----------------------------
def train_one(model, loader):
    model.train()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(EPOCHS_SHADOW):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()

shadow_models = []
member_logits, member_labels = [], []
nonmember_logits, nonmember_labels = [], []

shadow_size = 10000
shadow_test = 2000
for i in range(N_SHADOW):
    # disjoint slice
    start = i * (shadow_size + shadow_test)
    train_idx = mnist_indices[start : start + shadow_size]
    test_idx  = mnist_indices[start + shadow_size : start + shadow_size + shadow_test]
    train_set = Subset(mnist_train, train_idx)
    test_set  = Subset(mnist_train, test_idx)
    train_loader = DataLoader(train_set, batch_size=BATCH_SHADOW, shuffle=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SHADOW, shuffle=False)

    model = SimpleCNN().to(DEVICE)
    train_one(model, train_loader)
    shadow_models.append(model)

    # collect member logits (shadow train) and non-member logits (shadow test + external)
    def collect(loader, target):
        for x, y in loader:
            x = x.to(DEVICE)
            with torch.no_grad():
                logits = model(x).cpu()
            target.append(logits)
            target_labels.append(y)
    # member
    target_labels = member_labels
    collect(train_loader, member_logits)
    # non-member (in-dist)
    target_labels = nonmember_labels
    collect(test_loader, nonmember_logits)

# Extra non-member diversity
ext_loader = DataLoader(nonmember_ds, batch_size=BATCH_SHADOW, shuffle=False)
for x, y in ext_loader:
    x = x.to(DEVICE)
    with torch.no_grad():
        logits = shadow_models[0](x).cpu()   # any shadow for feature extraction
    nonmember_logits.append(logits)
    nonmember_labels.append(y)  # labels unused but keeps shape

# -----------------------------
# 6.  Build attack (feature, label) tensors
# -----------------------------
def build_features(logit_list):
    logits  = torch.cat(logit_list)
    probs   = F.softmax(logits, dim=1)
    max_prob, _ = probs.max(dim=1)
    ent     = softmax_entropy(logits)
    pred    = probs.argmax(dim=1)
    # pseudo-ground-truth for loss; use predicted label (black-box!)
    loss    = cross_entropy_loss(logits, pred)
    feats   = torch.stack([max_prob, ent, loss], dim=1)  # shape [N, 3]
    return feats.numpy()

X_member     = build_features(member_logits)
X_nonmember  = build_features(nonmember_logits)
y_member     = np.ones(len(X_member),      dtype=np.int64)
y_nonmember  = np.zeros(len(X_nonmember),  dtype=np.int64)

X_attack = np.concatenate([X_member, X_nonmember], axis=0)
y_attack = np.concatenate([y_member, y_nonmember], axis=0)

# Feature scaling to mimic StandardScaler → LightGBM
scaler = StandardScaler().fit(X_attack)
X_attack = scaler.transform(X_attack)

# Train / val split
perm = np.random.permutation(len(X_attack))
split = int(0.8 * len(X_attack))
train_idx, val_idx = perm[:split], perm[split:]
X_train, y_train = torch.tensor(X_attack[train_idx]).float(), torch.tensor(y_attack[train_idx])
X_val,   y_val   = torch.tensor(X_attack[val_idx]).float(), torch.tensor(y_attack[val_idx])

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_ATTACK, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=BATCH_ATTACK, shuffle=False)

# -----------------------------
# 7.  Attack classifier (simple MLP)
# -----------------------------
class AttackMLP(nn.Module):
    def __init__(self, in_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16), nn.ReLU(),
            nn.Linear(16, 8),  nn.ReLU(),
            nn.Linear(8, 2))
    def forward(self, x):  # returns logits
        return self.net(x)

attack_clf = AttackMLP().to(DEVICE)
opt = optim.Adam(attack_clf.parameters(), lr=5e-3)

for epoch in range(EPOCHS_ATTACK):
    attack_clf.train()
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        loss = F.cross_entropy(attack_clf(x), y)
        loss.backward()
        opt.step()

attack_clf.eval()
def eval_auc(loader):
    all_prob, all_y = [], []
    with torch.no_grad():
        for x, y in loader:
            out = attack_clf(x.to(DEVICE))
            prob = F.softmax(out, dim=1)[:, 1]
            all_prob.append(prob.cpu())
            all_y.append(y)
    all_prob = torch.cat(all_prob).numpy()
    all_y    = torch.cat(all_y).numpy()
    return roc_auc_score(all_y, all_prob), accuracy_score(all_y, (all_prob>0.5))

auc, acc = eval_auc(val_loader)
print(f"[Attack-Val]  AUC = {auc:.4f} | ACC = {acc:.4f}")

# -----------------------------
# 8.  Using the attack on the real target model
# -----------------------------
def target_model_api(x):
    """
    Black-box stub – replace this with real API queries.
    Here we just reuse a trained shadow model as a stand-in.
    """
    with torch.no_grad():
        return shadow_models[0](x.to(DEVICE)).cpu()

def inference_attack(data_loader):
    attack_probs = []
    for x, _ in data_loader:
        logits = target_model_api(x)
        feats  = scaler.transform(build_features([logits]))
        feats  = torch.tensor(feats).float().to(DEVICE)
        with torch.no_grad():
            proba = F.softmax(attack_clf(feats), dim=1)[:, 1]
        attack_probs.extend(proba.cpu().numpy())
    return np.array(attack_probs)

# Example – evaluate privacy risk on MNIST *train* (members) vs *test* (non-members)
mnist_full = datasets.MNIST(root="./data", train=True,  download=False, transform=transform)
mnist_test = datasets.MNIST(root="./data", train=False, download=False, transform=transform)

mem_loader  = DataLoader(Subset(mnist_full, np.arange(2048)), batch_size=256)
non_loader  = DataLoader(Subset(mnist_test, np.arange(2048)), batch_size=256)

mem_scores  = inference_attack(mem_loader)
non_scores  = inference_attack(non_loader)
labels      = np.concatenate([np.ones_like(mem_scores), np.zeros_like(non_scores)])
scores      = np.concatenate([mem_scores,               non_scores])
auc_target  = roc_auc_score(labels, scores)
print(f"[Target-Model]  AUC = {auc_target:.4f}  → privacy risk estimate")
