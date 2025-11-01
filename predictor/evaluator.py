import os
import argparse
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision import models
from tqdm import tqdm
from torch.utils.data import DataLoader
from posture.OpenPoseKeras.pose_init import pose_init

from .dataloader import get_preprocessing, BabySleepCocoDataset


def create_dataloaders(base_dir, split="test", batch_size=32, num_workers=0, transform_fn=get_preprocessing):
    split_dir = os.path.join(base_dir, split)
    ann_path = os.path.join(split_dir, "_annotations.coco.json")

    dataset = BabySleepCocoDataset(split_dir, ann_path, transform=transform_fn())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader


def evaluate_model(base_dir, model_path, batch_size=1, num_workers=0, num_classes=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = create_dataloaders(base_dir, split="test", batch_size=batch_size, num_workers=num_workers)

    # ✅ Load DenseNet for 10-class
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluating"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # ✅ Multi-class metrics
    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec  = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    print("\n✅ Test Set Evaluation Results")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Baby Pose Classification Model")
    parser.add_argument("--base_dir", type=str, default="dataset")
    parser.add_argument("--model_path", type=str, default="best_model_densenet.pth")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--weights", type=str, help="Path to pose estimator weights")

    args = parser.parse_args()
    pose_init(args.weights)

    evaluate_model(
        base_dir=args.base_dir,
        model_path=args.model_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
