"""
Inference script for testing model on individual images
"""

import torch
from torchvision import transforms
from PIL import Image
import argparse
from model import CarComponentCNN
import os


def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint"""
    
    model = CarComponentCNN()
    print("Initializing CarComponentCNN")
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"âœ… Loaded model from {checkpoint_path}")
        print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"   Val Acc: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    else:
        print(f"âš ï¸  Warning: Checkpoint not found at {checkpoint_path}")
        print("   Using untrained model")
    
    model = model.to(device)
    model.eval()
    
    return model


def predict_image(image_path, model, device='cuda', threshold=0.5):
    """
    Run inference on a single image
    
    Args:
        image_path: Path to image file
        model: Loaded model
        device: Device to run inference on
        threshold: Classification threshold
        
    Returns:
        dict: Component predictions
    """
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = outputs[0].cpu().numpy()
        predictions = (outputs > threshold).int()[0].cpu().numpy()
    
    # Format results
    component_names = ['Front Left', 'Front Right', 'Rear Left', 'Rear Right', 'Hood']
    results = {}
    
    for i, name in enumerate(component_names):
        state = 'Open' if predictions[i] == 1 else 'Closed'
        confidence = probabilities[i] if predictions[i] == 1 else (1 - probabilities[i])
        results[name] = {
            'state': state,
            'confidence': confidence,
            'probability': probabilities[i]
        }
    
    return results


def print_results(results):
    """Pretty print prediction results"""
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    
    for component, data in results.items():
        state = data['state']
        confidence = data['confidence']
        prob = data['probability']
        
        # Color coding for terminal
        state_marker = "ðŸŸ¢" if state == 'Open' else "ðŸ”´"
        
        print(f"\n{state_marker} {component:15s}: {state:7s}")
        print(f"   Confidence: {confidence*100:5.2f}%")
        print(f"   Probability: {prob:.4f}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='Run inference on car images')
    parser.add_argument('image', type=str, help='Path to input image')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold (default: 0.5)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for inference')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("="*60)
    print("CAR COMPONENT INFERENCE")
    print("="*60)
    print(f"Image: {args.image}")
    print(f"Device: {device}")
    print(f"Threshold: {args.threshold}")
    print("="*60)
    
    # Check image exists
    if not os.path.exists(args.image):
        print(f"\nâŒ Error: Image not found at {args.image}")
        return
    
    # Load model
    print("\nLoading model...")
    model = load_model(args.checkpoint, device=device)
    
    # Run inference
    print(f"\nRunning inference on {args.image}...")
    results = predict_image(args.image, model, device=device, threshold=args.threshold)
    
    # Print results
    print_results(results)
    
    # Summary
    open_components = [name for name, data in results.items() if data['state'] == 'Open']
    closed_components = [name for name, data in results.items() if data['state'] == 'Closed']
    
    print("\nSUMMARY:")
    print(f"  Open: {len(open_components)} component(s)")
    if open_components:
        print(f"    - {', '.join(open_components)}")
    print(f"  Closed: {len(closed_components)} component(s)")
    if closed_components:
        print(f"    - {', '.join(closed_components)}")


if __name__ == '__main__':
    main()
