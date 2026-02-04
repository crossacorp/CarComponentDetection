import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from model import CarComponentCNN


class CarComponentDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = dataset_dir
        self.images_dir = os.path.join(dataset_dir, "images")
        self.labels_dir = os.path.join(dataset_dir, "labels")
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(self.images_dir) if f.endswith('.png')])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        label_name = img_name.replace('.png', '.json')
        label_path = os.path.join(self.labels_dir, label_name)
        
        with open(label_path, 'r') as f:
            label_data = json.load(f)
        
        components = label_data['components']
        label_vector = [
            1 if components['front_left'] == 'open' else 0,
            1 if components['front_right'] == 'open' else 0,
            1 if components['rear_left'] == 'open' else 0,
            1 if components['rear_right'] == 'open' else 0,
            1 if components['hood'] == 'open' else 0
        ]
        
        label = torch.tensor(label_vector, dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_train_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


def get_val_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


def train():
    DATASET_DIR = "dataset"
    CHECKPOINT_DIR = "checkpoints"
    BATCH_SIZE = 32
    VAL_SPLIT = 0.2
    NUM_EPOCHS = 1000
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 1e-4
    PATIENCE = 10
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {device}")
    print(f"Loading dataset from {DATASET_DIR}...")
    
    train_transform = get_train_transform()
    val_transform = get_val_transform()
    
    full_dataset = CarComponentDataset(DATASET_DIR, transform=None)
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    model = CarComponentCNN()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.15
    
    model = model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    best_val_acc = 0.0
    patience_counter = 0
    
    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    print("="*70)
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
        for images, labels in train_bar:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predictions = (outputs > 0.5).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.numel()
            
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predictions = (outputs > 0.5).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.numel()
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1:3d} | Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.2f}%", end='')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc,
                'val_acc': val_acc,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }
            torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
            print(" [saved]")
        else:
            patience_counter += 1
            print()
        
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    print("="*70)
    print(f"Training complete. Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to {CHECKPOINT_DIR}/best_model.pth")


if __name__ == "__main__":
    train()
