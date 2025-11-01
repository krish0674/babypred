import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
from dataloader import BabySleepCocoDataset, get_train_augs, get_preprocessing
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

def create_dataloaders(base_dir, batch_size=32, num_workers=0):

    def make_loader(split, transform_fn, shuffle):
        split_dir = os.path.join(base_dir, split)
        ann_path = os.path.join(split_dir, "_annotations.coco.json")
        dataset = BabySleepCocoDataset(split_dir, ann_path, transform=transform_fn())
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    train_loader = make_loader("train", get_train_augs, True)
    val_loader = make_loader("valid", get_preprocessing, False)
    test_loader = make_loader("test", get_preprocessing, False)

    return train_loader, val_loader, test_loader


def train_model(base_dir="dataset", batch_size=32, num_epochs=10, lr=1e-4, weight_decay=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _ = create_dataloaders(base_dir, batch_size)

    # ✅ Visualize first few samples
    save_dir = "./vis_samples"
    os.makedirs(save_dir, exist_ok=True)

    class_names = [str(i) for i in range(10)]  # OR load names list

    count = 0
    for imgs, labels in train_loader:
        for i in range(imgs.shape[0]):
            img = imgs[i].permute(1, 2, 0).cpu().numpy()
            label = labels[i].item()

            plt.figure()
            plt.imshow(img)
            plt.title(f"Class {class_names[label]}")
            plt.axis("off")

            file_path = os.path.join(save_dir, f"sample_{count}_class_{label}.png")
            plt.savefig(file_path, bbox_inches='tight')
            plt.close()

            count += 1
            if count == 3:
                break
        break

    print(f"Saved sample visualizations to: {save_dir}")

    # ✅ Model
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, 10)  # ✅ Multi-class
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # ✅ Train
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

        # ✅ Validation
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

        # ✅ Save best by accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model_densenet_10classes.pth")
            print("✅ Best model saved")

    print(f"\n✅ Best Validation Accuracy: {best_acc:.4f}")
