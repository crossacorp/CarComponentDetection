"""
Download Qwen2-VL-2B-Instruct model from HuggingFace
Run this once before training.
"""

from huggingface_hub import snapshot_download
import os

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
LOCAL_DIR = "./model_cache/Qwen2-VL-2B-Instruct"

def main():
    print("=" * 60)
    print("DOWNLOADING Qwen2-VL-2B-Instruct")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Save to: {LOCAL_DIR}")
    print("This may take a few minutes (~5GB download)...")
    print("=" * 60)

    os.makedirs(LOCAL_DIR, exist_ok=True)

    path = snapshot_download(
        repo_id=MODEL_NAME,
        local_dir=LOCAL_DIR,
    )

    print(f"\nâœ… Model downloaded successfully to: {path}")
    print("=" * 60)
    print("\nNext step: run `python prepare_dataset.py`")


if __name__ == "__main__":
    main()
