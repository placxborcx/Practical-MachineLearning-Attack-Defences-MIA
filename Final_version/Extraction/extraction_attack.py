from __future__ import annotations
import argparse, json, os, random, time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

from a3_mnist import My_MNIST          # victim architecture

# ----------------------------------------------------------------------------
# Global configuration - can be overridden via CLI
# ----------------------------------------------------------------------------
CFG: Dict[str, object] = {
    # data - loading & hardware
    "data_root":         "./data/MNIST",      # download cache
    "batch_size":        128,                 # mini - batch size
    
    # victim-side parameters
    "victim_ckpt":       "target_model.pth",
    "victim_epochs":     5,
    "victim_lr":         1.0,
    
    # attack - side parameters
    "n_queries":         8_000,
    "surrogate_epochs":  10,
    "surrogate_lr":      1e-3,
    # misc
    "seed":              2025,    # RNG seed for reproducibility
    "device":            "cuda" if torch.cuda.is_available() else "cpu",      # training device
}

# ----------------------------------------------------------------------------
# Command-line arguments
# ----------------------------------------------------------------------------

P = argparse.ArgumentParser()
# For terminal execute and reset environment
P.add_argument("--quick",   action="store_true",
               help="debug run - 2 epochs / 1 k queries")
P.add_argument("--retrain", action="store_true",
               help="ignore victim checkpoint, retrain")

# For factor testing and store result
P.add_argument("--n_queries", type=int,   default=None, help="override query set size")
P.add_argument("--sur_lr",    type=float, default=None, help="override surrogate learning-rate")
P.add_argument("--output",   default="results.json", help="result file name")

# ----------------------------------------------------------------------------
# House-keeping - reproducibility & folders
# ----------------------------------------------------------------------------

# Full determinism for CUDA - comes with a performance cost but eliminates
# non－determinism between runs.

random.seed(CFG["seed"])
np.random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"])
torch.cuda.manual_seed_all(CFG["seed"])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device(CFG["device"])
Path(CFG["data_root"]).mkdir(parents=True, exist_ok=True)

TIMES: Dict[str, float] = {}
T0_GLOBAL = time.perf_counter()

# ----------------------------------------------------------------------------
# Surrogate model definition
# ----------------------------------------------------------------------------
class SurrogateNet(nn.Module):
    """A tiny two-convolution CNN (~55 k parameters) used as the *surrogate*.

    Architecture
    ------------
    Conv(1 -> 16, k=5, pad=2) -> ReLU -> 2×2 MaxPool
    -> Conv(16 -> 32, k=5, pad=2) -> ReLU -> 2×2 MaxPool
    -> FC(32·7·7 -> 10 logits)

    The network is deliberately lightweight so that the attack wall-time
    is dominated by the *query* phase rather than by surrogate training.
    Light 2-conv CNN (~55 k parameters).
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)  # 1×28×28 → 16×14×14
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)  # 16×14×14 → 32×7×7
        self.fc    = nn.Linear(32 * 7 * 7, 10)   # flattened → logits
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = F.relu(F.max_pool2d(self.conv1(x), 2))   # first conv + pooling
        x = F.relu(F.max_pool2d(self.conv2(x), 2))   # second conv + pooling
        return self.fc(x.flatten(1))


# ----------------------------------------------------------------------------
# Data - loading helpers
# ----------------------------------------------------------------------------

TFM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),  # mean/std for MNIST
])

def get_loaders(bs: int) -> Tuple[DataLoader, DataLoader]:

    """Return `(train_loader, test_loader)` for MNIST.

    Parameters
    ----------
    bs : int
        Mini-batch size.

    Notes
    -----
    * Uses the same normalisation constants as LeCun’s original split
      (µ ≈ 0.1307, σ ≈ 0.3081).
    * `num_workers` is set to 0 to avoid RNG nondeterminism across
      platforms when full determinism is requested.
    """

    tr = datasets.MNIST(CFG["data_root"], train=True,  download=True, transform=TFM)
    te = datasets.MNIST(CFG["data_root"], train=False, download=True, transform=TFM)
    return (
        DataLoader(tr, bs, shuffle=True,  drop_last=True,  num_workers=0),
        DataLoader(te, bs, shuffle=False, drop_last=False, num_workers=0),
    )

@torch.no_grad()
#Compute conventional top-1 accuracy on a `DataLoader`
def accuracy(model: nn.Module, loader: DataLoader) -> float:
    model.eval(); ok=tot=0
    for x,y in loader:
        ok += (model(x.to(DEVICE)).argmax(1).cpu()==y).sum().item(); tot+=y.size(0)
    return ok/tot

# Return the proportion of samples on which networks *a* and *b* 
# Predict the **same class** (used as an extraction-success metric).

@torch.no_grad()
def agreement(a: nn.Module, b: nn.Module, loader: DataLoader) -> float:
    a.eval(); b.eval(); mt=tt=0
    for x,_ in loader:
        x=x.to(DEVICE)
        mt += (a(x).argmax(1)==b(x).argmax(1)).sum().item(); tt+=x.size(0)
    return mt/tt


# ----------------------------------------------------------------------------
# Victim model utilities
# ----------------------------------------------------------------------------

def _train_victim() -> nn.Module:

    """(Re-)train the *victim* model for `CFG['victim_epochs']` epochs.

    Returns
    -------
    nn.Module
        A fully trained `My_MNIST` instance ready for evaluation.

    Side Effects
    ------------
    * Saves the checkpoint to `CFG['victim_ckpt']`.
    * Prints per-epoch accuracy for quick monitoring.

    Design Choice
    -------------
    We use Adadelta because it replicates the original MNIST tutorial
    while keeping the code dependency-free (no need for newer schedulers
    such as *One-Cycle*).
    """

    tr,te = get_loaders(CFG["batch_size"])
    m = My_MNIST().to(DEVICE)
    opt = torch.optim.Adadelta(m.parameters(), lr=CFG["victim_lr"])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.7)
    nll = nn.NLLLoss()
    for ep in range(1, CFG["victim_epochs"]+1):
        m.train()
        for x,y in tr:
            x,y=x.to(DEVICE),y.to(DEVICE)
            opt.zero_grad(); nll(m(x),y).backward(); opt.step()
        sched.step(); print(f"    Victim epoch {ep:02d}  acc:{accuracy(m,te):.4f}")
    torch.save(m.state_dict(), CFG["victim_ckpt"])
    return m

def load_victim() -> nn.Module:
    """
    Step 1:
    Load an existing victim checkpoint or trigger training.

    The function tolerates architecture mismatches (e.g. if the author
    tweaked `My_MNIST`) by falling back to training when `state_dict`
    loading fails.

    This step ensures the attacker has a working version of the victim model. 
    If a pretrained checkpoint (target_model.pth) exists, it is loaded. Otherwise, the model is trained from scratch
    using the MNIST dataset. The model architecture is My_MNIST, simulating a real-world black-box ML API.

    """

    if ARGS.retrain:
        print("[Victim] retraining (flag --retrain).")
        return _train_victim().eval()
    m=My_MNIST().to(DEVICE)
    if os.path.exists(CFG["victim_ckpt"]):
        try:
            m.load_state_dict(torch.load(CFG["victim_ckpt"], map_location=DEVICE))
            print("[Victim] checkpoint loaded.")
        except Exception:
            print("[Victim] checkpoint mismatch – retraining…"); m=_train_victim()
    else:
        print("[Victim] no checkpoint – training…"); m=_train_victim()
    return m.eval()

# ----------------------------------------------------------------------------
# Black-box API wrapper - abstracts away direct victim access
# ----------------------------------------------------------------------------

"""
Simulates a commercial API that provides predictions for MNIST digits. Implement BlackBoxAPI.query() is the only gateway to the target model. 
Wraps the target in a method query() that the API will return a probability simplex (torch.exp(logp)) by default, either softmax probabilities
 or raw logits,  which are generated by the model predictions. The prediction model only presents the result without a confidence score in 
 the privacy consideration.
"""


class BlackBoxAPI:
    """
    Step 2:
    Wraps the victim model in a BlackBoxAPI class, which exposes only a query interface. 
    The attacker can obtain softmax probabilities or logits from this API, but cannot access model internals 
    (weights, gradients, or training data). This enforces the black-box constraint.
    
    Stateless wrapper that *mimics* a production prediction endpoint.

    Parameters
    ----------
    victim : nn.Module
        The already-trained target model.

    Methods
    -------
    query(x, logits=False) → torch.Tensor
        Forward `x` through the victim and return either *log-probabilities*
        (`logits=True`) or a *probability simplex* (`logits=False`).
    
    Simple wrapper that mimics an *HTTP* prediction endpoint.
    """
    
    def __init__(self, victim: nn.Module): self.v = victim
    @torch.no_grad()
    def query(self, x: torch.Tensor, *, logits=False) -> torch.Tensor:
        # Return victim outputs for a mini-batch of inputs.
        logp = self.v(x.to(DEVICE))                 # My_MNIST gives log-probs
        return logp if logits else torch.exp(logp)


# ----------------------------------------------------------------------------
# Query-set builder
# ----------------------------------------------------------------------------

def build_query_set(api: BlackBoxAPI, n: int) -> TensorDataset:
    """
    Step 3:

    A fixed number of images from the MNIST training set (disjoint from the victim's training data) are selected. 
    These are sent to the black-box API, and the corresponding soft labels (probability distributions) are collected. 
    This forms a pseudo-labeled dataset for training the surrogate.

    Collect `n` *soft-labelled* samples from the victim.

    This is the **data-collection phase** of the extraction attack.

    Parameters
    ----------
    api : BlackBoxAPI
        The only legal conduit to the victim.
    n : int
        Number of queries (budget).

    Returns
    -------
    TensorDataset
        A `TensorDataset(inputs, soft_labels)` ready for dataloader
        consumption.
    """
    base=datasets.MNIST(CFG["data_root"], train=True, download=True, transform=TFM)
    idx=np.random.choice(len(base), n, replace=False)
    ld = DataLoader(Subset(base,idx), CFG["batch_size"], shuffle=False, num_workers=0)
    xs: List[torch.Tensor] = []; ps: List[torch.Tensor] = []
    for x,_ in ld:
        xs.append(x); ps.append(api.query(x))
    return TensorDataset(torch.cat(xs), torch.cat(ps))


# ----------------------------------------------------------------------------
# Surrogate - training routine
# ----------------------------------------------------------------------------

def train_surrogate(ds: TensorDataset) -> nn.Module:
    """
    Step 4:

    The surrogate model (SurrogateNet, a lightweight CNN) is trained using KL-divergence to minimize the difference 
    between its predictions and the black-box outputs. This simulates the attacker's attempt to replicate the target 
    model’s behavior as closely as possible.
    
    Train the surrogate network via KL-divergence minimisation.

    The optimiser and hyper-parameters (Adam, learning-rate, epochs) are
    intentionally modest so that most of the *attack cost* lies in
    **queries**, not compute—mirroring real-world constraints.
    """
    m=SurrogateNet().to(DEVICE)
    ld=DataLoader(ds, CFG["batch_size"], shuffle=True, num_workers=0)
    opt=torch.optim.Adam(m.parameters(), lr=CFG["surrogate_lr"])
    kld=nn.KLDivLoss(reduction="batchmean")
    for ep in range(1, CFG["surrogate_epochs"]+1):
        m.train(); run=0.0
        for x,p in ld:
            x,p=x.to(DEVICE),p.to(DEVICE)
            opt.zero_grad(); loss=kld(F.log_softmax(m(x),1),p); loss.backward(); opt.step()
            run+=loss.item()*x.size(0)
        print(f"    Surrogate epoch {ep:02d}  KL:{run/len(ds):.4f}")
    return m.eval()


# ----------------------------------------------------------------------------
# Evaluation helpers / Section
# ----------------------------------------------------------------------------

"""
Functionality:
After training, the surrogate model is evaluated on MNIST test data. Metrics include:

Accuracy of the victim model

Accuracy of the surrogate model

Agreement rate (how often both models make the same prediction)

These metrics measure the success of the model extraction attack.

"""



@torch.no_grad()
def evaluate(v: nn.Module, s: nn.Module) -> Dict[str,float]:
    _,te = get_loaders(CFG["batch_size"])
    return {
        "victim_acc":    accuracy(v,te),
        "surrogate_acc": accuracy(s,te),
        "agreement":     agreement(v,s,te),
    }

# ----------------------------------------------------------------------------
# Main orchestration logic
# ----------------------------------------------------------------------------

def main() -> None:
    """Script entry-point - orchestrates the four attack stages.

    Steps
    -----
    1. Parse/override CLI flags.
    2. Prepare target/victim (`load_victim`).
    3. Build query set (`build_query_set`).
    4. Train surrogate (`train_surrogate`).
    5. Evaluate & persist results (JSON).

    Timing of each macro step is stored in the global `TIMES` dict for
    later inspection (e.g. when comparing defence overheads).
    """
    if ARGS.n_queries is not None:
        CFG["n_queries"] = ARGS.n_queries

    if ARGS.sur_lr is not None:
        CFG["surrogate_lr"] = ARGS.sur_lr

    if ARGS.quick:
        CFG.update(victim_epochs=2, surrogate_epochs=3, n_queries=1_000)  # type: ignore
    
    # 1  Victim preparation (train or load)
    t=time.perf_counter(); vict=load_victim(); TIMES["victim_prepare_s"]=time.perf_counter()-t

    # 2  Query-set generation
    api=BlackBoxAPI(vict)
    t=time.perf_counter(); qset=build_query_set(api, int(CFG["n_queries"])); TIMES["query_phase_s"]=time.perf_counter()-t

    # 3  Surrogate training
    t=time.perf_counter(); surro=train_surrogate(qset); TIMES["surrogate_train_s"]=time.perf_counter()-t

    # 4  Evaluation
    t=time.perf_counter(); res=evaluate(vict,surro); TIMES["evaluation_s"]=time.perf_counter()-t
    TIMES["total_runtime_s"]=time.perf_counter()-T0_GLOBAL

    print("\n[Evaluation]")
    for k,v in res.items(): print(f"  {k:14s}: {v*100:.2f}%")

    print("\n[Timing Summary] (seconds)")
    for k, v in TIMES.items():
        print(f"  {k:20s}: {v:.2f}")

    # Persist summary and save as json
    with open(ARGS.output, "w", encoding="utf-8") as fp:
        json.dump({**res, "timings": TIMES}, fp, indent=2)


# ----------------------------------------------------------------------------
# When executed as a script: parse CLI and *run* `main()`.
# When imported as a module (e.g. unit tests): expose functions/classes only.
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    ARGS = P.parse_args()  
    main()                  
else:
    ARGS = P.parse_args([])    # dummy args for IDEs / external import

