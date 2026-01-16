import torch
import sys
from safetensors.torch import save_file

model_name, new_name = sys.argv[1:]

splat = new_new.split('.')
assert splat[-1] == 'safetensors'

# Load your legacy .pt checkpoint
checkpoint = torch.load(model_name, weights_only=True)

# Separate weights from other info
weights = checkpoint["model_state_dict"]

# Save as safetensors
save_file(weights, new_name)
