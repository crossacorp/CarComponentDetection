"""
Objective 2: Vision Language Model Description
Minimal production-ready web interface
"""

import os
import io
import base64
import torch
from flask import Flask, render_template_string, jsonify, request
from flask_cors import CORS
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    BitsAndBytesConfig,
)
from peft import PeftModel
from utils import crop_to_car

app = Flask(__name__)
CORS(app)

# Config
BASE_MODEL_PATH = "./model_cache/Qwen2-VL-2B-Instruct"
LORA_PATH = "./vlm_checkpoints/best_lora"

# Globals
model = None
processor = None
device = None

INSTRUCTION = (
    "Look at this car image carefully. "
    "Describe the status of each component: "
    "front left door, front right door, rear left door, rear right door, and hood. "
    "State whether each one is open or closed."
)


def load_model():
    """Load base model + LoRA adapter"""
    global model, processor, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH)

    print("Loading base model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, LORA_PATH)
    model.eval()

    print("Model loaded successfully!")


def predict(image: Image.Image) -> str:
    """Run inference: image -> description text"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": INSTRUCTION},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    model_inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=1.0,
        )

    generated_ids = output_ids[0][model_inputs["input_ids"].shape[1]:]
    response = processor.decode(generated_ids, skip_special_tokens=True)

    return response


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VLM Car Description</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }
        .container { max-width: 700px; margin: 0 auto; }
        .header {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .header h1 { font-size: 24px; margin-bottom: 8px; }
        .header p { color: #666; font-size: 14px; }
        .card {
            background: white;
            padding: 24px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .button-group {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-bottom: 20px;
        }
        button {
            padding: 12px 20px;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
        }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        button:hover:not(:disabled) { transform: translateY(-1px); }
        .btn-primary { background: #007bff; color: white; }
        .btn-primary:hover:not(:disabled) { background: #0056b3; }
        .btn-secondary { background: #28a745; color: white; }
        .btn-secondary:hover:not(:disabled) { background: #218838; }
        .error {
            background: #f8d7da;
            color: #721c24;
            padding: 12px;
            border-radius: 6px;
            margin-bottom: 16px;
            display: none;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #007bff;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 12px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .preview-section {
            margin-bottom: 20px;
            display: none;
        }
        .preview-image {
            width: 100%;
            max-height: 300px;
            object-fit: contain;
            border-radius: 8px;
            background: #f8f9fa;
            border: 2px solid #e0e0e0;
        }
        .result-section { display: none; }
        .description-box {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            line-height: 1.7;
            color: #333;
            font-size: 15px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>VLM Car Description</h1>
            <p>Objective 2 - Fine-tuned Vision Language Model</p>
        </div>

        <div class="card">
            <div class="button-group">
                <button class="btn-primary" id="openCarBtn">Open Car Model</button>
                <button class="btn-secondary" id="captureBtn">Capture & Describe</button>
            </div>

            <div class="error" id="error"></div>
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Analyzing with VLM...</p>
            </div>

            <div class="preview-section" id="previewSection">
                <img id="capturedImage" class="preview-image" alt="Captured car">
            </div>

            <div class="result-section" id="resultSection">
                <h3 style="margin-bottom: 12px; color: #333;">Description</h3>
                <div class="description-box" id="descriptionBox"></div>
            </div>
        </div>
    </div>

    <script>
        const CAR_URL = 'https://euphonious-concha-ab5c5d.netlify.app/';
        let carWindow = null;
        let mediaStream = null;
        let videoElement = null;

        const openCarBtn = document.getElementById('openCarBtn');
        const captureBtn = document.getElementById('captureBtn');
        const loading = document.getElementById('loading');
        const error = document.getElementById('error');
        const previewSection = document.getElementById('previewSection');
        const capturedImage = document.getElementById('capturedImage');
        const resultSection = document.getElementById('resultSection');
        const descriptionBox = document.getElementById('descriptionBox');

        openCarBtn.onclick = () => {
            if (carWindow && !carWindow.closed) {
                carWindow.focus();
                return;
            }
            carWindow = window.open(CAR_URL, 'carModel', 'width=1200,height=800');
            if (!carWindow) {
                showError('Please allow popups for this site.');
            }
        };

        async function initializeCaptureStream() {
            if (mediaStream && mediaStream.active) return true;
            try {
                mediaStream = await navigator.mediaDevices.getDisplayMedia({
                    video: { width: { ideal: 1920 }, height: { ideal: 1080 } }
                });
                if (!videoElement) {
                    videoElement = document.createElement('video');
                    videoElement.autoplay = true;
                    videoElement.style.display = 'none';
                    document.body.appendChild(videoElement);
                }
                videoElement.srcObject = mediaStream;
                await new Promise(resolve => { videoElement.onloadedmetadata = resolve; });
                mediaStream.getVideoTracks()[0].onended = () => { mediaStream = null; };
                return true;
            } catch (err) {
                showError('Screen capture error: ' + err.message);
                return false;
            }
        }

        async function captureAndDescribe() {
            loading.style.display = 'block';
            resultSection.style.display = 'none';
            previewSection.style.display = 'none';
            showError('');
            captureBtn.disabled = true;

            try {
                if (!await initializeCaptureStream()) return;
                await new Promise(resolve => setTimeout(resolve, 100));

                const canvas = document.createElement('canvas');
                canvas.width = videoElement.videoWidth;
                canvas.height = videoElement.videoHeight;
                canvas.getContext('2d').drawImage(videoElement, 0, 0);
                const imageData = canvas.toDataURL('image/png');

                const response = await fetch('http://localhost:5000/describe', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: imageData })
                });

                if (!response.ok) throw new Error(`Server error: ${response.status}`);
                const data = await response.json();
                
                if (data.error) {
                    showError(data.error);
                } else {
                    if (data.cropped_image) {
                        capturedImage.src = data.cropped_image;
                        previewSection.style.display = 'block';
                    }
                    descriptionBox.textContent = data.description;
                    resultSection.style.display = 'block';
                }
            } catch (err) {
                showError(err.message.includes('Failed to fetch')
                    ? 'Cannot connect to backend. Ensure Flask app is running.'
                    : 'Error: ' + err.message);
            } finally {
                loading.style.display = 'none';
                captureBtn.disabled = false;
            }
        }

        function showError(message) {
            if (message) {
                error.textContent = message;
                error.style.display = 'block';
            } else {
                error.style.display = 'none';
            }
        }

        captureBtn.onclick = captureAndDescribe;

        window.addEventListener('beforeunload', () => {
            if (carWindow && !carWindow.closed) carWindow.close();
            if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
        });
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/describe', methods=['POST'])
def describe():
    """Receive image, run VLM inference, return description"""
    try:
        data = request.get_json()
        image_data = data.get('image')

        if not image_data:
            return jsonify({"error": "No image data provided"}), 400

        # Decode base64 image
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]

        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Crop to car region
        image = crop_to_car(image)

        # Convert cropped image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        cropped_base64 = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Run VLM inference
        description = predict(image)

        return jsonify({
            "description": description,
            "cropped_image": cropped_base64,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "model_loaded": model is not None,
        "device": str(device) if device else None,
    })


if __name__ == '__main__':
    print("="*60)
    print("OBJECTIVE 2: VLM CAR DESCRIPTION")
    print("="*60)

    load_model()

    print("\nStarting server on http://localhost:5000")
    print("="*60)
    app.run(host='0.0.0.0', port=5000, debug=False)
