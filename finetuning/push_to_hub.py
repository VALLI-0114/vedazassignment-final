"""
push_to_hub.py
---------------
Uploads the trained LoRA adapter (recommended — small, fast) or the merged
full model to the Hugging Face Hub so it can be linked in the submission.

Prereqs:
    pip install huggingface_hub
    huggingface-cli login          # or export HF_TOKEN=...

Usage:
    # Push just the LoRA adapter (few MB, recommended)
    python push_to_hub.py --local_dir out/qwen2.5-vedaz-lora \
                           --repo_id <your-hf-username>/vedaz-qwen2.5-lora

    # Push the merged full model instead
    python push_to_hub.py --local_dir out/qwen2.5-vedaz-lora-merged \
                           --repo_id <your-hf-username>/vedaz-qwen2.5-merged
"""
import argparse
from huggingface_hub import HfApi, create_repo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_dir", required=True)
    ap.add_argument("--repo_id", required=True, help="e.g. username/vedaz-qwen2.5-lora")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    api = HfApi()
    create_repo(args.repo_id, private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=args.local_dir,
        repo_id=args.repo_id,
        repo_type="model",
    )
    print(f"Uploaded. Model card / files live at: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
