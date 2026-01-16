from huggingface_hub import HfApi
import sys

model_path, model_name = sys.argv[1:]

api = HfApi()

# This creates the repo if it doesn't exist and uploads the weights
api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo="model.safetensors",
    repo_id=model_name,
    repo_type="model"
)
