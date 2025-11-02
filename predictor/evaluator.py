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
    split_dir = os.path.join(base_dir, split)
    ann_path = os.path.join(split_dir, "_annotations.coco.json")
    dataset = BabySleepCocoDataset(split_dir, ann_path, split=split, transform=transform_fn())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader


def draw_text_on_image(image, actual, predicted, prob, correct=True):
    """Draw actual and predicted labels with probability on the given PIL image."""
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()

    # Green if correct, red if incorrect
    color = "green" if correct else "red"

    text1 = f"Actual: {actual}"
    text2 = f"Predicted: {predicted} ({prob:.2f})"

    # Background rectangle
    draw.rectangle([(0, 0), (image.width, 70)], fill=(0, 0, 0))
    draw.text((10, 10), text1, fill="white", font=font)
    draw.text((10, 40), text2, fill=color, font=font)

    return image


def evaluate_model(base_dir, model_path, batch_size=1, num_workers=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_loader = create_dataloaders(base_dir, split="test", batch_size=batch_size, num_workers=num_workers)

    # Load pretrained DenseNet121
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds, all_labels = [], []

    # Folder to save annotated predictions
    save_dir = "predictions"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    class_map = {0: "safe", 1: "unsafe"}

    print("\nStarting Evaluation... (0: safe, 1: unsafe)")

    img_counter = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Handle datasets that return (imgs, labels) or (imgs, labels, paths)
            if len(batch) == 3:
                imgs, labels, paths = batch
            else:
                imgs, labels = batch
                paths = [None] * len(imgs)

            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for i in range(len(imgs)):
                actual_class = class_map[int(labels[i].cpu().numpy())]
                predicted_class = class_map[int(preds[i].cpu().numpy())]
                prob = float(confs[i].cpu().numpy())
                correct = (actual_class == predicted_class)

                # Try to load the original image if path is available
                if paths[i] is not None and os.path.exists(paths[i]):
                    try:
                        original_img = Image.open(paths[i]).convert("RGB")
                    except Exception as e:
                        print(f"Error loading image {paths[i]}: {e}")
                        continue
                else:
                    # Generate placeholder image if no path provided
                    img_array = (imgs[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    original_img = Image.fromarray(img_array)

                # Annotate and save
                annotated_img = draw_text_on_image(original_img, actual_class, predicted_class, prob, correct)
                img_counter += 1
                save_path = os.path.join(save_dir, f"img_{img_counter:04d}.jpg")
                annotated_img.save(save_path)

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"\n--- Test Set Evaluation Results ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"\nAnnotated images saved in: {os.path.abspath(save_dir)}")


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
