import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
from .dataloader import BabySleepCocoDataset,get_train_augs,get_preprocessing
from torch.utils.data import Dataset,DataLoader
import os 
import matplotlib.pyplot as plt

def create_dataloaders(base_dir, batch_size=32, num_workers=0):
    """
    base_dir/
      ├── train/
      │   ├── _annotations.coco.json
      │   ├── image1.jpg ...
      ├── val/
      │   ├── _annotations.coco.json
      ├── test/
          ├── _annotations.coco.json
    """

    def make_loader(split, transform_fn, shuffle,limit):
        split_dir = os.path.join(base_dir, split)
        ann_path = os.path.join(split_dir, "_annotations.coco.json")
        dataset = BabySleepCocoDataset(split_dir, ann_path, transform=transform_fn(),limit=limit)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    train_loader = make_loader("train", get_train_augs, True,limit=5)
    val_loader = make_loader("valid", get_preprocessing, False,limit=2)
    test_loader = make_loader("test", get_preprocessing, False,limit=1)

    return train_loader, val_loader,test_loader

def train_model(base_dir="dataset" ,batch_size=32,
                num_epochs=10, lr=1e-4, weight_decay=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _ = create_dataloaders(
        base_dir=base_dir,
        batch_size=batch_size
    )
    
    for i, (img, label) in enumerate(train_loader):
        if i == 3: break
        plt.imshow(img.squeeze().permute(1,2,0).numpy())
        plt.title("Safe" if label.item() == 1 else "Unsafe")
        plt.axis("off")
        plt.show()

    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, 2)  
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_acc = float('inf')

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        model.train()
        train_loss, train_corrects, total = 0.0, 0, 0
        for imgs, labels in tqdm(train_loader, desc="Training"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            train_loss += loss.item() * imgs.size(0)
            train_corrects += torch.sum(preds == labels).item()
            total += labels.size(0)

        train_loss /= total
        train_acc = train_corrects / total

        model.eval()
        val_loss, val_corrects, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc="Validation"):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * imgs.size(0)
                val_corrects += torch.sum(preds == labels).item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_corrects / val_total

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        if val_loss < best_acc:
            best_acc = val_loss
            torch.save(model.state_dict(), "best_model_densenet.pth")
            print("✅ Best model saved")

    print(f"\nBest Validation Accuracy: {best_acc:.4f}")


