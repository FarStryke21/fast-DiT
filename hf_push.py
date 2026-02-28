from huggingface_hub import HfApi, create_repo

def push_checkpoint_to_hub():
    api = HfApi()
    
    # 1. Define your repo name
    repo_id = "FarStryke21/cmu-10799-dit-b2" 
    
    # 2. Create the repository (if it doesn't already exist)
    print(f"Creating repository {repo_id}...")
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    
    # 3. Upload the checkpoint file
    local_file_path = "results/000-DiT-B-2/checkpoints/final-checkpoint.pt"
    hub_file_path = "final-checkpoint.pt"
    
    print(f"Uploading {local_file_path} to the Hub...")
    api.upload_file(
        path_or_fileobj=local_file_path,
        path_in_repo=hub_file_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Uploading final EMA checkpoint for DiT-B/2"
    )
    
    print(f"Success! Model weights are now live at: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    push_checkpoint_to_hub()