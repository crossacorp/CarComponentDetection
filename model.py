import torch
import torch.nn as nn
import torch.nn.functional as F

class CarComponentCNN(nn.Module):
    """
    Optimized CNN for 3,200 samples.
    Reduces FC layer size to prevent overfitting while maintaining 
    spatial feature extraction.
    """
    
    def __init__(self, num_components=5):
        super(CarComponentCNN, self).__init__()
        
        # Block 1: 224x224 -> 112x112
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 2: 112x112 -> 56x56
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 3: 56x56 -> 28x28
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Spatial preservation: 28x28 -> 7x7
        self.adaptive_pool = nn.AdaptiveMaxPool2d((7, 7))
        
        # Optimized Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(128 * 7 * 7, 128), 
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(128, num_components),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.adaptive_pool(x)
        
        # Flatten: (Batch, 128, 7, 7) -> (Batch, 6272)
        x = x.view(x.size(0), -1)
        
        x = self.fc(x)
        return x

    def predict_components(self, x, threshold=0.5):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            predictions = (outputs > threshold).int()
            
            component_names = ['front_left', 'front_right', 'rear_left', 'rear_right', 'hood']
            results = {}
            
            for i, name in enumerate(component_names):
                results[name] = 'open' if predictions[0][i].item() == 1 else 'closed'
            
            return results

if __name__ == "__main__":
    model = CarComponentCNN()
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel output shape: {output.shape}")
    print(f"Expected shape: (1, 5)")
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"\nWith 1,280 training samples:")
    print(f"  Samples per parameter: {1280/total_params:.1f}")
    print(f"  Target range: 10-100 samples per parameter")
    

    # Test prediction
    predictions = model.predict_components(dummy_input)
    print(f"\nSample predictions: {predictions}")
    
    # Show layer sizes
    print("\n" + "="*60)
    print("LAYER ARCHITECTURE:")
    print("="*60)
    for name, param in model.named_parameters():
        print(f"{name:30s} : {str(param.shape):20s} | {param.numel():6,} params")
    print("="*60)
    
    # Dropout analysis
    print("\nDropout Layers:")
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            print(f"  {name}: p={module.p}")
