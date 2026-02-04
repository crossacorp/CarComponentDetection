"""
Objective 3: Visual Grounding Model
Minimal production-ready web interface using GroundingDINO
"""

import os
import io
import base64
import tempfile
import torch
import numpy as np
from flask import Flask, render_template_string, jsonify, request
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont
from utils import crop_to_car

# GroundingDINO imports
try:
    from groundingdino.util.inference import load_model, load_image, predict
    GROUNDINGDINO_AVAILABLE = True
except ImportError:
    GROUNDINGDINO_AVAILABLE = False
    print("GroundingDINO not installed!")

app = Flask(__name__)
CORS(app)

# Config
MODEL_CONFIG = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
MODEL_CHECKPOINT = "groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.25

# Globals
model = None
device = None

COLORS = [
    (255, 107, 107),
    (78, 205, 196),
    (69, 183, 209),
    (255, 160, 122),
    (152, 216, 200),
]


def download_grounding_dino_model():
    """Download GroundingDINO model if not exists"""
    import urllib.request
    
    if not os.path.exists(MODEL_CHECKPOINT):
        print("Downloading GroundingDINO model...")
        url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
        urllib.request.urlretrieve(url, MODEL_CHECKPOINT)
    
    os.makedirs("groundingdino/config", exist_ok=True)
    if not os.path.exists(MODEL_CONFIG):
        url = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        urllib.request.urlretrieve(url, MODEL_CONFIG)


def load_grounding_model():
    """Load GroundingDINO model"""
    global model, device
    
    if not GROUNDINGDINO_AVAILABLE:
        raise ImportError("GroundingDINO not installed. Run: pip install groundingdino-py")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    download_grounding_dino_model()
    model = load_model(MODEL_CONFIG, MODEL_CHECKPOINT, device=str(device))
    print("Model loaded successfully!")


def parse_text_prompt(text_prompt):
    """Parse text prompt into GroundingDINO format"""
    text_prompt = text_prompt.lower().strip()
    
    for prefix in ["locate the ", "locate ", "find the ", "find ", "show me ", "show "]:
        text_prompt = text_prompt.replace(prefix, "")
    
    if "hood" in text_prompt or "bonnet" in text_prompt:
        return "car hood . engine hood"
    elif "door" in text_prompt:
        if "all" in text_prompt:
            return "car door . front door . rear door"
        elif "front left" in text_prompt:
            return "front left car door . car door"
        elif "front right" in text_prompt:
            return "front right car door . car door"
        elif "rear left" in text_prompt or "back left" in text_prompt:
            return "rear left car door . car door"
        elif "rear right" in text_prompt or "back right" in text_prompt:
            return "rear right car door . car door"
        elif "front" in text_prompt:
            return "front car door . car door"
        elif "rear" in text_prompt or "back" in text_prompt:
            return "rear car door . car door"
        else:
            return "car door . vehicle door"
    
    return text_prompt


def filter_detections(predictions, image_width, image_height):
    """Filter out bad detections using NMS"""
    from torchvision.ops import nms
    
    if len(predictions['bboxes']) == 0:
        return predictions
    
    image_area = image_width * image_height
    filtered_predictions = {'bboxes': [], 'labels': [], 'scores': []}
    
    # Filter by size and confidence
    for bbox, label, score in zip(predictions['bboxes'], predictions['labels'], predictions['scores']):
        x1, y1, x2, y2 = bbox
        box_area = (x2 - x1) * (y2 - y1)
        area_ratio = box_area / image_area
        
        if area_ratio > 0.75 or area_ratio < 0.01 or score < 0.30:
            continue
        
        filtered_predictions['bboxes'].append(bbox)
        filtered_predictions['labels'].append(label)
        filtered_predictions['scores'].append(score)
    
    if len(filtered_predictions['bboxes']) == 0:
        return filtered_predictions
    
    # Apply NMS
    boxes_tensor = torch.tensor(filtered_predictions['bboxes'], dtype=torch.float32)
    scores_tensor = torch.tensor(filtered_predictions['scores'], dtype=torch.float32)
    keep_indices = nms(boxes_tensor, scores_tensor, 0.5)
    
    return {
        'bboxes': [filtered_predictions['bboxes'][i] for i in keep_indices],
        'labels': [filtered_predictions['labels'][i] for i in keep_indices],
        'scores': [filtered_predictions['scores'][i] for i in keep_indices]
    }


def run_grounding(image: Image.Image, text_prompt: str):
    """Run GroundingDINO on the image"""
    grounding_caption = parse_text_prompt(text_prompt)
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
        temp_path = tmp_file.name
        image.save(temp_path)
    
    try:
        _, image_tensor = load_image(temp_path)
        
        boxes, logits, phrases = predict(
            model=model,
            image=image_tensor,
            caption=grounding_caption,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=str(device)
        )
        
        h, w = np.array(image).shape[:2]
        boxes_pixel = boxes * torch.Tensor([w, h, w, h])
        
        predictions = {'bboxes': [], 'labels': [], 'scores': []}
        
        for box, score, phrase in zip(boxes_pixel, logits, phrases):
            cx, cy, w_box, h_box = box.tolist()
            x1 = cx - w_box / 2
            y1 = cy - h_box / 2
            x2 = cx + w_box / 2
            y2 = cy + h_box / 2
            
            predictions['bboxes'].append([x1, y1, x2, y2])
            predictions['labels'].append(phrase)
            predictions['scores'].append(float(score))
        
        predictions = filter_detections(predictions, w, h)
        return predictions
        
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


def draw_bounding_boxes(image: Image.Image, predictions: dict):
    """Draw bounding boxes on the image"""
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        except:
            font = ImageFont.load_default()
    
    for idx, (bbox, label, score) in enumerate(zip(predictions['bboxes'], predictions['labels'], predictions['scores'])):
        color = COLORS[idx % len(COLORS)]
        color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=color_hex, width=4)
        
        label_text = f"{label} ({score:.2f})"
        text_bbox = draw.textbbox((x1, y1 - 28), label_text, font=font)
        text_bg = [text_bbox[0] - 4, text_bbox[1] - 4, text_bbox[2] + 4, text_bbox[3] + 4]
        draw.rectangle(text_bg, fill=color_hex)
        draw.text((x1, y1 - 28), label_text, fill="white", font=font)
    
    return image


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visual Grounding</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }
        .container { max-width: 800px; margin: 0 auto; }
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
        .btn-generate {
            background: #6f42c1;
            color: white;
            grid-column: 1 / -1;
        }
        .btn-generate:hover:not(:disabled) { background: #5a32a3; }
        .input-group { margin-bottom: 20px; }
        .input-group label {
            display: block;
            margin-bottom: 8px;
            color: #333;
            font-weight: 500;
        }
        .input-group input {
            width: 100%;
            padding: 10px 14px;
            border: 2px solid #e0e0e0;
            border-radius: 6px;
            font-size: 14px;
        }
        .input-group input:focus {
            outline: none;
            border-color: #007bff;
        }
        .example-prompts {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 8px;
        }
        .example-prompt {
            background: #f0f0f0;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .example-prompt:hover {
            background: #007bff;
            color: white;
        }
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
        .result-section { display: none; }
        .result-images {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }
        .image-container { text-align: center; }
        .image-label {
            font-weight: 500;
            color: #666;
            margin-bottom: 8px;
            font-size: 13px;
        }
        .result-image {
            width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        @media (max-width: 768px) {
            .result-images { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Visual Grounding</h1>
            <p>Objective 3 - GroundingDINO Component Localization</p>
        </div>

        <div class="card">
            <div class="button-group">
                <button class="btn-primary" id="openCarBtn">Open Car Model</button>
                <button class="btn-secondary" id="captureBtn">Capture Screenshot</button>
            </div>

            <div class="input-group">
                <label for="promptInput">Text Prompt</label>
                <input 
                    type="text" 
                    id="promptInput" 
                    placeholder="e.g., locate the open doors"
                    value="locate the open doors"
                >
                <div class="example-prompts">
                    <div class="example-prompt" data-prompt="locate the open doors">open doors</div>
                    <div class="example-prompt" data-prompt="locate the front left door">front left</div>
                    <div class="example-prompt" data-prompt="locate the hood">hood</div>
                    <div class="example-prompt" data-prompt="locate all doors">all doors</div>
                </div>
            </div>

            <button class="btn-generate" id="generateBtn" disabled>Generate Grounding</button>

            <div class="error" id="error"></div>
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Running GroundingDINO...</p>
            </div>

            <div class="result-section" id="resultSection">
                <h3 style="margin-bottom: 16px; color: #333;">Results</h3>
                <div class="result-images">
                    <div class="image-container">
                        <div class="image-label">Original</div>
                        <img id="originalImage" class="result-image" alt="Original">
                    </div>
                    <div class="image-container">
                        <div class="image-label">Grounded</div>
                        <img id="groundedImage" class="result-image" alt="Grounded">
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const CAR_URL = 'https://euphonious-concha-ab5c5d.netlify.app/';
        let carWindow = null;
        let mediaStream = null;
        let videoElement = null;
        let capturedImageData = null;

        const openCarBtn = document.getElementById('openCarBtn');
        const captureBtn = document.getElementById('captureBtn');
        const generateBtn = document.getElementById('generateBtn');
        const promptInput = document.getElementById('promptInput');
        const loading = document.getElementById('loading');
        const error = document.getElementById('error');
        const resultSection = document.getElementById('resultSection');
        const originalImage = document.getElementById('originalImage');
        const groundedImage = document.getElementById('groundedImage');

        openCarBtn.onclick = () => {
            if (carWindow && !carWindow.closed) {
                carWindow.focus();
                return;
            }
            carWindow = window.open(CAR_URL, 'carModel', 'width=1200,height=800');
            if (!carWindow) showError('Please allow popups for this site.');
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

        async function captureScreenshot() {
            captureBtn.disabled = true;
            showError('');
            try {
                if (!await initializeCaptureStream()) return;
                await new Promise(resolve => setTimeout(resolve, 100));
                const canvas = document.createElement('canvas');
                canvas.width = videoElement.videoWidth;
                canvas.height = videoElement.videoHeight;
                canvas.getContext('2d').drawImage(videoElement, 0, 0);
                capturedImageData = canvas.toDataURL('image/png');
                generateBtn.disabled = false;
                showError('Screenshot captured successfully');
                error.style.background = '#d4edda';
                error.style.color = '#155724';
                error.style.display = 'block';
            } catch (err) {
                showError('Error: ' + err.message);
            } finally {
                captureBtn.disabled = false;
            }
        }

        async function generateGrounding() {
            if (!capturedImageData) {
                showError('Please capture a screenshot first');
                return;
            }
            const prompt = promptInput.value.trim();
            if (!prompt) {
                showError('Please enter a text prompt');
                return;
            }

            loading.style.display = 'block';
            resultSection.style.display = 'none';
            showError('');
            generateBtn.disabled = true;

            try {
                const response = await fetch('http://localhost:5001/ground', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: capturedImageData, prompt: prompt })
                });

                if (!response.ok) throw new Error(`Server error: ${response.status}`);
                const data = await response.json();
                
                if (data.error) {
                    showError(data.error);
                } else {
                    originalImage.src = data.original_image;
                    groundedImage.src = data.grounded_image;
                    resultSection.style.display = 'block';
                }
            } catch (err) {
                showError(err.message.includes('Failed to fetch')
                    ? 'Cannot connect to backend. Ensure Flask app is running.'
                    : 'Error: ' + err.message);
            } finally {
                loading.style.display = 'none';
                generateBtn.disabled = false;
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

        document.querySelectorAll('.example-prompt').forEach(btn => {
            btn.onclick = () => { promptInput.value = btn.dataset.prompt; };
        });

        captureBtn.onclick = captureScreenshot;
        generateBtn.onclick = generateGrounding;

        promptInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !generateBtn.disabled) generateGrounding();
        });

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


@app.route('/ground', methods=['POST'])
def ground():
    """Receive image and text prompt, run grounding, return image with bounding boxes"""
    try:
        data = request.get_json()
        image_data = data.get('image')
        text_prompt = data.get('prompt')

        if not image_data:
            return jsonify({"error": "No image data provided"}), 400
        if not text_prompt:
            return jsonify({"error": "No text prompt provided"}), 400

        # Decode base64 image
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]

        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Crop to car region
        original_image = crop_to_car(image)
        
        # Convert original to base64
        buffered = io.BytesIO()
        original_image.save(buffered, format="PNG")
        original_base64 = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Run grounding
        predictions = run_grounding(original_image, text_prompt)
        
        # Draw bounding boxes
        grounded_image = original_image.copy()
        grounded_image = draw_bounding_boxes(grounded_image, predictions)
        
        # Convert grounded image to base64
        buffered = io.BytesIO()
        grounded_image.save(buffered, format="PNG")
        grounded_base64 = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({
            "original_image": original_base64,
            "grounded_image": grounded_base64,
            "num_detections": len(predictions['bboxes'])
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "model_loaded": model is not None,
        "groundingdino_available": GROUNDINGDINO_AVAILABLE
    })


if __name__ == '__main__':
    print("="*60)
    print("OBJECTIVE 3: VISUAL GROUNDING")
    print("="*60)

    if not GROUNDINGDINO_AVAILABLE:
        print("\nERROR: GroundingDINO not installed!")
        print("Install with: pip install groundingdino-py")
        print("="*60)
        exit(1)

    load_grounding_model()

    print("\nStarting server on http://localhost:5001")
    print("="*60)
    app.run(host='0.0.0.0', port=5001, debug=False)
