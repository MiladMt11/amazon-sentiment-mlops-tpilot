from huggingface_hub import create_repo, upload_folder
import os

# Define HF username and desired repo name
HF_USERNAME = "your HF username"
REPO_NAME = "sentiment-api-model"
LOCAL_MODEL_DIR = "model_3epochs"

# Full repo ID
repo_id = f"{HF_USERNAME}/{REPO_NAME}"

# Create the repo (only once)
create_repo(repo_id, repo_type="model", private=False, exist_ok=True)

# Upload the model folder
upload_folder(
    repo_id=repo_id,
    folder_path=LOCAL_MODEL_DIR,
    path_in_repo=".",
    repo_type="model"
)

print(f"Model uploaded to https://huggingface.co/{repo_id}")
