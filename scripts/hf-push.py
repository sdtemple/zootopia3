import torch
import sys
from zootopia3 import MyCNN  # Ensure this inherits from PyTorchModelHubMixin

model_name, model_repo = sys.argv[1:]

# 1. Load the legacy checkpoint
checkpoint = torch.load(model_name, weights_only=True, map_location="cpu")

# 2. Extract configuration directly from the checkpoint
# Assuming your training script saved it under a "config" or similar key
config_params = checkpoint['config']

# 3. Initialize model with the exact parameters from the checkpoint
model = MyCNN(**config_params)

# 4. Load the weights
model.load_state_dict(checkpoint["model_state_dict"])

# 5. Push everything to the Hub
# The mixin automatically creates 'config.json' from your config_params
model.push_to_hub(model_repo, config=config_params)
