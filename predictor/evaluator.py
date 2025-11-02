import os
import argparse
import torch
import shutil
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision import models
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from posture.OpenPoseKeras.pose_init import pose_init
from .dataloader import get_preprocessing, BabySleepCocoDataset


def create_dataloaders(base_dir, split="test", batch_size=32, num_workers=0, transform_fn=get_preprocessing):
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
    split_dir = os.path.join(base_dir, split)
    ann_path = os.path.join(split_dir, "_annotations.coco.json")
    dataset = BabySleepCocoDataset(split_dir, ann_path, split=split, transform=transform_fn())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader


def draw_text_on_image(image, actual, predicted, prob):
    """
    Draws actual and predicted text with probability on the given PIL image.
    """
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()

    text1 = f"Actual: {actual}"
    text2 = f"Predicted: {predicted} ({prob:.2f})"

    # Background rectangles for better visibility
    draw.rectangle([(0, 0), (image.width, 70)], fill=(0, 0, 0, 128))
    draw.text((10, 10), text1, fill="white", font=font)
    draw.text((10, 40), text2, fill="white", font=font)

    return image


def evaluate_model(base_dir, model_path, batch_size=1, num_workers=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_loader = create_dataloaders(base_dir, split="test", batch_size=batch_size, num_workers=num_workers)

    # Load pretrained DenseNet121 and modify final layer
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds, all_labels = [], []

    # Folder to save predictions
    save_dir = "predictions"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    class_map = {0: "safe", 1: "unsafe"}

    print("\nStarting Evaluation... (0: safe, 1: unsafe)")

    with torch.no_grad():
        for idx, (imgs, labels, paths) in enumerate(tqdm(test_loader, desc="Evaluating")):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for i in range(len(imgs)):
                img_path = paths[i]
                img_name = os.path.basename(img_path)
                actual_class = class_map[int(labels[i].cpu().numpy())]
                predicted_class = class_map[int(preds[i].cpu().numpy())]
                prob = float(confs[i].cpu().numpy())

                # Load original image
                try:
                    original_img = Image.open(img_path).convert("RGB")
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
                    continue

                # Draw results on the image
                annotated_img = draw_text_on_image(original_img, actual_class, predicted_class, prob)

                # Save image
                save_path = os.path.join(save_dir, img_name)
                annotated_img.save(save_path)

    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"\n--- Test Set Evaluation Results ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"\nAnnotated predictions saved to: {os.path.abspath(save_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Baby Sleep Classification Model")
    parser.add_argument("--base_dir", type=str, default="dataset", help="Base directory containing train/val/test folders")
    parser.add_argument("--model_path", type=str, default="best_model_densenet.pth", help="Path to the trained model")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of DataLoader workers")
    parser.add_argument("--weights", type=str, help="Weights for pose_init")

    args = parser.parse_args()
    pose_init(args.weights)

    evaluate_model(
        base_dir=args.base_dir,
        model_path=args.model_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
