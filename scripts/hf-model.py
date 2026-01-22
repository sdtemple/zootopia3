from huggingface_hub import HfApi
import torch
import sys
import json

model_path, hf_repo = sys.argv[1:]

# # You must hard code this
# # because I didn't correctly set up config.json in MyCNN class.
# # It should be fixed on January 22, 2026.
# config_data = {
#     "model_type": "custom_pytorch_model",
#     "num_classes": 4, # for shape model
#     # "num_classes": 8, # for color model
#     "height": 224,
#     "width": 224,
#     "num_input_channels": 3,
#     "num_cnn_channels": 16,
#     "num_cnn_layers": 3,
#     "hidden_dim": 16,
#     "num_layers": 1,
#     "kernel_size": 3,
#     "stride": 1,
#     "padding": 1,
#     "pooling": 2,
#     "dropout": 0.2
# }

# with open("config.json", "w") as f:
#     json.dump(config_data, f)

api = HfApi()

# This creates the repo if it doesn't exist and uploads the weights
api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo="model.safetensors",
    repo_id=hf_repo,
    repo_type="model",
)

# api.upload_file(
#     path_or_fileobj="config.json",
#     path_in_repo="config.json",
#     repo_id=hf_repo,
#     repo_type="model",
#     commit_message="Add config.json to enable download tracking"
# )
