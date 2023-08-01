import os
import torch
from PIL import Image
import json
import alpr.ai_utils.metrics as metrics
import numpy as np

class InstanceSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, images: list, transforms):
        self.transforms = transforms
        self.images = images

    def __getitem__(self, idx) -> torch.Tensor:
        img = self.transforms(self.images[idx])
        return img

    def __len__(self):
        return len(self.images)


class InstanceSegmentationCocoDataset(torch.utils.data.Dataset):
    def __init__(self, root, images_path, coco_labels_path, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(self.root + images_path)))
        with open(self.root + coco_labels_path, "r") as f:
            self.labels_dict = json.load(f)

        # sort images to match order of labels
        self.ordered_images = []
        for u in range(len(self.labels_dict["images"])):
            for file in self.imgs:
                _path = self.labels_dict["images"][u]["file_name"].split("/")
                if len(_path) > 1:
                    file_name = _path[-1]
                elif len(_path) == 1:
                    file_name = _path
                else:
                    raise FileNotFoundError
                if file == file_name:
                    image = Image.open(f"{root}{images_path}{file_name}").convert("RGB")
                    self.ordered_images.append(image)
                    break

        targets = [{} for _ in range(len(self.imgs))]
        for ann in self.labels_dict["annotations"]:
            boxes = []
            # in coco format first 2 values are coords of left up point
            # second 2 values and width and height of box
            for j in range(0, len(ann["bbox"]), 4):
                x1, y1 = float(ann["bbox"][j]), float(ann["bbox"][j + 1])
                width, height = float(ann["bbox"][j + 2]), float(ann["bbox"][j + 3])

                x2, y2 = x1 + width, y1 + height
                boxes.append([x1, y1, x2, y2])

            target = {}
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
            # labels starts from 1; 0 is background
            target["labels"] = torch.as_tensor(
                [ann["category_id"] + 1], dtype=torch.int64
            )
            target["image_id"] = torch.tensor([ann["image_id"]], dtype=torch.int64)
            target["area"] = torch.as_tensor([ann["area"]], dtype=torch.float32)
            target["iscrowd"] = torch.as_tensor([ann["iscrowd"]], dtype=torch.int64)
            if len(targets[ann["image_id"]]) == 0:
                
                targets[ann["image_id"]] = target
            else:
                for key, value in target.items():
                    targets[ann["image_id"]][key] = torch.cat((targets[ann["image_id"]][key], value), 0)
        
        for i, t in enumerate(targets):
            if len(t) < 1:
                target = {}
                target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
                target["labels"] = torch.zeros(0, dtype=torch.int64)
                target["image_id"] = torch.tensor(i, dtype=torch.int64)
                target["area"] = torch.zeros(0, dtype=torch.float32)
                target["iscrowd"] = torch.zeros(0, dtype=torch.int64)
                targets[i] = target
        self.targets = targets

    def merge_dataset(self, dataset):
        """merge diffrent instances of this dataset class"""
        self.ordered_images.append(dataset.ordered_images)
        self.targets.append(dataset.targets)

    def __getitem__(self, idx):
        if self.transforms is not None:
            img, target = self.transforms(self.ordered_images[idx], self.targets[idx])
        else:
            img = self.ordered_images[idx]
            target = self.targets[idx]

        return img, target

    def __len__(self):
        return len(self.imgs)