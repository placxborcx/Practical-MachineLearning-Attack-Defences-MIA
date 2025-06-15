import torch
from membership_inference_attack import SimpleCNN, Config      # Read original structure
from MIA_defense_mechanism import DefenceWrapper                   # read defense

# 1) load attack original model
base = SimpleCNN().cpu()
base.load_state_dict(torch.load("target_final.pth", map_location="cpu"))

# 2) apply defense
wrapped = DefenceWrapper(base)      # run all defense

# 3) save as new file，for defense_test.py 
torch.save(base.state_dict(), "defended_target.pth")
print("✔  defended_target.pth saved (plain weights)")
