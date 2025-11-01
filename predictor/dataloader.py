import os
import json
import cv2
import torch
from torch.utils.data import Dataset
from albumentations import (
    Compose, Resize, Normalize, HorizontalFlip, RandomBrightnessContrast,
    ShiftScaleRotate, CoarseDropout
)
from albumentations.pytorch import ToTensorV2
from posture.OpenPoseKeras.pose_init import pose_process

def get_train_augs():
    return Compose([
        HorizontalFlip(p=0.5),
        RandomBrightnessContrast(p=0.3),
        ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        #CoarseDropout(max_holes=1, max_height=32, max_width=32, fill_value=0, p=0.3),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_preprocessing():
    return Compose([
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

class BabySleepCocoDataset(Dataset):
    def __init__(self, images_dir, annotation_path, transform=None):
        self.images_dir = images_dir
        self.pose_dir = os.path.join(images_dir, "processed_pose")  # new folder
        os.makedirs(self.pose_dir, exist_ok=True)

        self.transform = transform

        with open(annotation_path, 'r') as f:
            data = json.load(f)

        self.image_id_to_name = {img['id']: img['file_name'] for img in data['images']}
        self.image_to_label = {ann['image_id']: ann['category_id'] for ann in data['annotations']}
        self.categories = {c['id']: c['name'] for c in data['categories']}

        safe_classes = {"baby_on_back"}
        self.cat_to_binary = {
            cid: 1 if self.categories[cid] in safe_classes else 0
            for cid in self.categories
        }

        self.samples = [
            (self.image_id_to_name[iid], self.cat_to_binary[self.image_to_label[iid]])
            for iid in self.image_id_to_name if iid in self.image_to_label
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]

        src_img_path = os.path.join(self.images_dir, img_name)
        
        # pose image path
        pose_img_path = os.path.join(self.pose_dir, img_name.replace(".jpg", "_pose.jpg"))

        # If pose saved already, load it
        if os.path.exists(pose_img_path):
            image = cv2.imread(pose_img_path)
        else:
            # run pose only once
            image = pose_process(src_img_path)
            if image is None:
                raise ValueError(f"Pose failed: {src_img_path}")

            cv2.imwrite(pose_img_path, image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)['image']

        return image, torch.tensor(label, dtype=torch.long)
