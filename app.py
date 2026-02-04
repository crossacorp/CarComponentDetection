"""
Objective 1: Real-time Car Component Detection
Minimal production-ready web interface
"""

from flask import Flask, render_template_string, jsonify, request
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
import io
import base64
from model import CarComponentCNN
import os
from utils import preprocess_image

app = Flask(__name__)
CORS(app)

# Global model
model = None
device = None
transform = None


def load_model(checkpoint_path='checkpoints/best_model.pth'):
    """Load the trained model"""
    global model, device, transform
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = CarComponentCNN()
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded: Val Acc {checkpoint.get('val_acc', 'N/A'):.2f}%")
    else:
        print(f"Warning: No checkpoint found at {checkpoint_path}")
    
    model = model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


def predict_from_image(image_data):
    """Run inference on image"""
    global model, device, transform
    
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        # Convert base64 to PIL Image
        if isinstance(image_data, str):
            if 'base64,' in image_data:
                image_data = image_data.split('base64,')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        else:
            image = image_data
        
        # Preprocess (crop to car)
        cropped_image = preprocess_image(image)
        
        # Convert cropped image to base64 for display
        buffered = io.BytesIO()
        cropped_image.save(buffered, format="PNG")
        cropped_base64 = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Transform for model
        image_tensor = transform(cropped_image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = outputs[0].cpu().numpy()
            predictions = (outputs > 0.5).int()[0]
        
        # Format results
        component_names = ['front_left', 'front_right', 'rear_left', 'rear_right', 'hood']
        component_labels = ['Front Left', 'Front Right', 'Rear Left', 'Rear Right', 'Hood']
        
        results = {
            'cropped_image': cropped_base64
        }
        
        for i, (name, label) in enumerate(zip(component_names, component_labels)):
            prob = float(probabilities[i])
            state = 'Open' if predictions[i].item() == 1 else 'Closed'
            confidence = prob if state == 'Open' else (1 - prob)
            
            results[label] = {
                'state': state,
                'confidence': confidence,
                'probability': prob
            }
        
        return results
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Car Component Detection</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }
        .container { max-width: 600px; margin: 0 auto; }
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
        .status-grid {
            display: grid;
            gap: 12px;
        }
        .status-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 16px;
            background: #f8f9fa;
            border-radius: 6px;
        }
        .status-label { font-weight: 500; color: #333; }
        .status-right {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .status-confidence {
            font-size: 13px;
            color: #666;
            font-weight: 500;
        }
        .status-badge {
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 13px;
            font-weight: 500;
        }
        .status-open { background: #d4edda; color: #155724; }
        .status-closed { background: #f8d7da; color: #721c24; }
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
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Car Component Detection</h1>
            <p>Objective 1 - Real-time CNN Detection</p>
        </div>

        <div class="card">
            <div class="button-group">
                <button class="btn-primary" id="openCarBtn">Open Car Model</button>
                <button class="btn-secondary" id="captureBtn">Capture & Detect</button>
            </div>

            <div class="error" id="error"></div>
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Analyzing...</p>
            </div>

            <div class="preview-section" id="previewSection">
                <h3 style="margin-bottom: 12px; color: #333;">Captured Image</h3>
                <img id="capturedImage" class="preview-image" alt="Captured car">
            </div>

            <div class="result-section" id="resultSection">
                <h3 style="margin-bottom: 16px; color: #333;">Component Status</h3>
                <div class="status-grid" id="statusGrid"></div>
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
        const statusGrid = document.getElementById('statusGrid');

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

        async function captureAndDetect() {
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

                const response = await fetch('http://localhost:5000/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: imageData })
                });

                if (!response.ok) throw new Error(`Server error: ${response.status}`);
                const data = await response.json();
                
                if (data.error) {
                    showError(data.error);
                } else {
                    displayResults(data);
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

        function displayResults(data) {
            // Display cropped image if available
            if (data.cropped_image) {
                capturedImage.src = data.cropped_image;
                previewSection.style.display = 'block';
            }
            
            statusGrid.innerHTML = '';
            
            // Remove cropped_image from display data
            const components = Object.keys(data).filter(key => key !== 'cropped_image');
            
            components.forEach(component => {
                const item = data[component];
                const div = document.createElement('div');
                div.className = 'status-item';
                
                const label = document.createElement('span');
                label.className = 'status-label';
                label.textContent = component;
                
                const rightDiv = document.createElement('div');
                rightDiv.className = 'status-right';
                
                const confidence = document.createElement('span');
                confidence.className = 'status-confidence';
                confidence.textContent = `${(item.confidence * 100).toFixed(1)}%`;
                
                const badge = document.createElement('span');
                badge.className = `status-badge status-${item.state.toLowerCase()}`;
                badge.textContent = item.state;
                
                rightDiv.appendChild(confidence);
                rightDiv.appendChild(badge);
                
                div.appendChild(label);
                div.appendChild(rightDiv);
                statusGrid.appendChild(div);
            });
            
            resultSection.style.display = 'block';
        }

        function showError(message) {
            if (message) {
                error.textContent = message;
                error.style.display = 'block';
            } else {
                error.style.display = 'none';
            }
        }

        captureBtn.onclick = captureAndDetect;

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


@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for predictions"""
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({"error": "No image data provided"}), 400
        
        results = predict_from_image(image_data)
        return jsonify(results)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "model_loaded": model is not None,
        "device": str(device) if device else "None"
    })


if __name__ == '__main__':
    print("="*60)
    print("OBJECTIVE 1: CAR COMPONENT DETECTION")
    print("="*60)
    
    CHECKPOINT_PATH = 'checkpoints/best_model.pth'
    load_model(CHECKPOINT_PATH)
    
    print("\nStarting server on http://localhost:5000")
    print("="*60)
    app.run(host='0.0.0.0', port=5000, debug=False)
