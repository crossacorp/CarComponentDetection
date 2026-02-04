"""
QLoRA Fine-tuning for Qwen2-VL-2B-Instruct
Optimized for 8GB VRAM (RTX 3070)

Uses:
- 4-bit quantization (QLoRA) to fit model in memory
- LoRA adapters on attention layers only
- Gradient accumulation to simulate larger batch sizes
"""

import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = "./model_cache/Qwen2-VL-2B-Instruct"
DATA_PATH = "./training_data.json"
OUTPUT_DIR = "./vlm_checkpoints"

# Training hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 1                  # 1 per GPU due to 8GB VRAM
GRADIENT_ACCUMULATION_STEPS = 4  # effective batch size = 4
LEARNING_RATE = 1e-4
MAX_NEW_TOKENS = 128

# LoRA config â€” higher rank + MLP layers = more capacity
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
# ============================================================


class CarVLMDataset(Dataset):
    """Dataset that loads images and pairs them with instruction/response."""

    def __init__(self, data_list, processor, max_length=1024):
        self.data = data_list
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load image
        image = Image.open(item["image_path"]).convert("RGB")

        # Build conversation in Qwen2-VL chat format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": item["instruction"]},
                ],
            },
            {
                "role": "assistant",
                "content": item["response"],
            },
        ]

        # Full text (input + response)
        text_full = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Input-only text (to find where response starts for masking)
        messages_input_only = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": item["instruction"]},
                ],
            },
        ]
        text_input = self.processor.apply_chat_template(
            messages_input_only, tokenize=False, add_generation_prompt=True
        )

        # Tokenize full sequence â€” this gives us pixel_values AND image_grid_thw
        model_inputs = self.processor(
            text=[text_full],
            images=[image],
            return_tensors="pt",
            padding=False,
        )

        # Tokenize input-only (text only, no image â€” just to get input token count)
        input_only_tokens = self.processor(
            text=[text_input],
            return_tensors="pt",
            padding=False,
        )

        input_ids = model_inputs["input_ids"].squeeze(0)
        attention_mask = model_inputs["attention_mask"].squeeze(0)
        pixel_values = model_inputs["pixel_values"]                # (num_patches, patch_dim)
        image_grid_thw = model_inputs["image_grid_thw"]            # (1, 3) â€” critical!

        # Create labels: -100 for input tokens, real ids for response tokens
        # Figure out response token count by comparing text-only tokenizations
        full_text_only_tokens = self.processor(
            text=[text_full],
            return_tensors="pt",
            padding=False,
        )
        input_len = input_only_tokens["input_ids"].shape[1]
        response_token_count = full_text_only_tokens["input_ids"].shape[1] - input_len
        mask_end = input_ids.shape[0] - response_token_count

        labels = input_ids.clone()
        labels[:mask_end] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }


def collate_fn(batch):
    """Custom collate: pad sequences and concatenate pixel_values / image_grid_thw."""
    max_len = max(item["input_ids"].shape[0] for item in batch)

    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    pixel_values_list = []
    image_grid_thw_list = []

    for item in batch:
        seq_len = item["input_ids"].shape[0]
        pad_len = max_len - seq_len

        input_ids_list.append(
            torch.cat([item["input_ids"], torch.full((pad_len,), 0, dtype=torch.long)])
        )
        attention_mask_list.append(
            torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        )
        labels_list.append(
            torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        )
        # pixel_values: each item is (num_patches, dim) â€” cat along dim 0
        pixel_values_list.append(item["pixel_values"])
        # image_grid_thw: each item is (1, 3) â€” cat along dim 0
        image_grid_thw_list.append(item["image_grid_thw"])

    return {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "labels": torch.stack(labels_list),
        "pixel_values": torch.cat(pixel_values_list, dim=0),
        "image_grid_thw": torch.cat(image_grid_thw_list, dim=0),
    }


def load_model_and_processor():
    """Load Qwen2-VL with 4-bit quantization and LoRA."""
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    print("Loading model in 4-bit (QLoRA)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Prepare for LoRA
    model = prepare_model_for_kbit_training(model)

    # LoRA config â€” target both attention AND MLP layers for more capacity
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",   # attention
            "gate_proj", "up_proj", "down_proj",        # MLP
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, processor


def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda")

    # Load data
    print("Loading dataset...")
    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    train_data = data["train"]
    val_data = data["val"]
    print(f"Train: {len(train_data)} | Val: {len(val_data)}")

    # Load model
    model, processor = load_model_and_processor()

    # Create datasets
    train_dataset = CarVLMDataset(train_data, processor)
    val_dataset = CarVLMDataset(val_data, processor)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    total_steps = (len(train_loader) // GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=50, num_training_steps=total_steps
    )

    # Training loop
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE} x {GRADIENT_ACCUMULATION_STEPS} accumulation = {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS} effective")
    print(f"Total steps: {total_steps}")
    print("=" * 60)

    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            image_grid_thw = batch["image_grid_thw"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

            loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()

            epoch_loss += outputs.loss.item()

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix({"loss": f"{outputs.loss.item():.4f}"})

        avg_train_loss = epoch_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                pixel_values = batch["pixel_values"].to(device)
                image_grid_thw = batch["image_grid_thw"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                )
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"\nEpoch {epoch+1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(OUTPUT_DIR, "best_lora")
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            print(f"  âœ… Saved best model (val_loss={avg_val_loss:.4f}) to {save_path}")

        # Save latest
        latest_path = os.path.join(OUTPUT_DIR, "latest_lora")
        model.save_pretrained(latest_path)
        processor.save_pretrained(latest_path)

    print("\n" + "=" * 60)
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {OUTPUT_DIR}/best_lora")
    print("=" * 60)
    print("\nNext step: run `python app_vlm.py` to launch the web interface")


if __name__ == "__main__":
    train()
