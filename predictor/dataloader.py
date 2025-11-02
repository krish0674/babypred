import os
import json
import cv2
import torch
from torch.utils.data import Dataset
from albumentations import (
    Compose, Resize, Normalize, HorizontalFlip, RandomBrightnessContrast,
    ShiftScaleRotate
)
from albumentations.pytorch import ToTensorV2

from posture.OpenPoseKeras.pose_init import pose_process


def get_train_augs():
    return Compose([
        HorizontalFlip(p=0.5),
        RandomBrightnessContrast(p=0.3),
        ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_preprocessing():
    return Compose([
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class BabySleepCocoDataset(Dataset):
    def __init__(self, images_dir, annotation_path, split="train",
                 transform=None, limit=None):

        self.images_dir = images_dir
        self.transform = transform

        # ✅ separate pose output folder for each split
        self.pose_dir = f"./pose_images_{split}"
        os.makedirs(self.pose_dir, exist_ok=True)

        with open(annotation_path, 'r') as f:
            data = json.load(f)

        self.image_id_to_name = {img['id']: img['file_name'] for img in data['images']}
        self.image_to_label = {ann['image_id']: ann['category_id'] for ann in data['annotations']}

        self.samples = [
            (os.path.join(images_dir, self.image_id_to_name[iid]),
             self.image_to_label[iid],
             self.image_id_to_name[iid])
            for iid in self.image_id_to_name if iid in self.image_to_label
        ]

        if limit:
            self.samples = self.samples[:limit]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, file_name = self.samples[idx]
        pose_img_path = os.path.join(self.pose_dir, file_name)

        # ✅ only generate pose once per image for this split
        if not os.path.exists(pose_img_path):
            pose_img = pose_process(img_path)
            if pose_img is None:
                raise ValueError(f"Pose failed for: {img_path}")
            cv2.imwrite(pose_img_path, pose_img)

        # ✅ load cached pose image
        image = cv2.imread(pose_img_path)
        if image is None:
            raise ValueError(f"Could not load pose result: {pose_img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)['image']

        return image, torch.tensor(label, dtype=torch.long)
