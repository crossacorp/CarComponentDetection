# Car Component Detection - Technical Test AI

Complete implementation of all three objectives:
- **Objective 1**: Custom CNN for real-time component detection
- **Objective 2**: Fine-tuned VLM for component description
- **Objective 3**: Visual grounding for component localization

## System Requirements

- Python 3.8+
- CUDA-capable GPU (8GB+ VRAM recommended)
- Google Chrome (for data collection)

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install GroundingDINO for Objective 3
pip install groundingdino-py
```

### 2. Objective 1: CNN Component Detection

**Data Collection**
```bash
python prepare_data.py
# Collects 1,600 images (32 states × 50 angles)
# Generates varied natural language descriptions
# Output: dataset/, training_data.json
```

**Dataset Preprocessing**
```bash
python reprocess_dataset.py
# Crops images to remove UI elements
# Output: dataset_cropped/
```

**Training**
```bash
python train.py
# Trains custom CNN
# Output: checkpoints/best_model.pth
```

**Launch Web Interface**
```bash
python app.py
# Access at: http://localhost:5000
```

### 3. Objective 2: Vision Language Model

**Download Base Model**
```bash
python download_model.py
# Downloads Qwen2-VL-2B-Instruct (~5GB)
```

**Fine-tune Model**
```bash
python train_vlm.py
# QLoRA fine-tuning (4-bit quantization)
# Output: vlm_checkpoints/best_lora
```

**Launch Web Interface**
```bash
python app_vlm.py
# Access at: http://localhost:5000
```

### 4. Objective 3: Visual Grounding (GroundingDINO)

**Launch Web Interface**
```bash
python app_grounding.py
# Access at: http://localhost:5001
```

## Project Structure

```
├── app.py                    # Objective 1 web interface
├── app_vlm.py                # Objective 2 web interface
├── app_grounding.py          # Objective 3 web interface (GroundingDINO)
├── utils.py                  # Shared utility functions
├── model.py                  # Custom CNN architecture
├── train.py                  # CNN training script
├── train_vlm.py              # VLM fine-tuning script
├── prepare_dataset.py        # dataset preparation
├── download_model.py         # Download Qwen2-VL model
├── inference.py              # Single image inference script
└── requirements.txt          # Python dependencies
```

## Model Architectures

### Objective 1: Custom CNN
- Lightweight architecture: ~400K parameters
- 3 convolutional blocks with batch normalization
- Adaptive pooling for spatial preservation
- Binary classification per component (5 outputs)

### Objective 2: Qwen2-VL-2B
- Base: Qwen2-VL-2B-Instruct (2B parameters)
- Fine-tuning: QLoRA (4-bit quantization)
- LoRA rank: 32, targeting attention + MLP layers
- Training: 5 epochs with gradient accumulation

### Objective 3: GroundingDINO
- State-of-the-art visual grounding model
- Text-to-image object localization

## Usage Examples

### Objective 1: Real-time Detection
1. Open http://localhost:5000
2. Click "Open Car Model"
3. Click "Capture & Detect" and select the car window
4. View real-time component status

### Objective 2: Natural Language Description
1. Open http://localhost:5000
2. Click "Open Car Model"
3. Click "Capture & Describe"
4. Read AI-generated description

### Objective 3: Text-based Grounding
1. Open http://localhost:5001
2. Click "Open Car Model"
3. Click "Capture Screenshot"
4. Enter text prompt (e.g., "locate the open doors")
5. View bounding boxes on components

## Development

### Test Single Image Inference
```bash
python inference.py path/to/image.png
```

### Adjust Model Hyperparameters
- CNN: Edit `train.py` (learning rate, epochs, dropout)
- VLM: Edit `train_vlm.py` (LoRA rank, batch size, epochs)
- Grounding: Edit `app_grounding.py` (BOX_THRESHOLD, TEXT_THRESHOLD)

## Technical Details

### Dataset
- 32 possible car states (2^5 combinations)
- 50 random camera angles per state
- Total: 1,600 training images
- Preprocessing: Auto-crop to remove UI, normalize

### Training Strategy
- CNN: Data augmentation (flip, rotation, color jitter)
- VLM: QLoRA for efficient fine-tuning
- Early stopping with patience
- Best model checkpointing

### Web Interface Technology
- Backend: Flask + Flask-CORS
- Frontend: Pure HTML/CSS/JavaScript
- Screen Capture API for real-time integration
- No external dependencies

## License

This project is developed for the Technical Test AI.
