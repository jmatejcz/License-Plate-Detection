import os
from PIL import Image
import numpy as np
import math
from collections import OrderedDict
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from alpr.ai_utils.transforms import EarlyStopping, AverageMeter, Eval, OCRLabelConverter
from tqdm import *
import torchvision.transforms.functional as F

class SynthCollator(object):
    def __call__(self, batch):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        width = [item["img"].shape[2] for item in batch]
        indexes = [item["idx"] for item in batch]
        imgs = torch.ones(
            [
                len(batch),
                batch[0]["img"].shape[0],
                batch[0]["img"].shape[1],
                max(width),
            ],
            dtype=torch.float32,
        ).to(device)
        for idx, item in enumerate(batch):
            try:
                imgs[idx, :, :, 0 : item["img"].shape[2]] = item["img"]
            except:
                print(imgs.shape)
        item = {"img": imgs, "idx": indexes}
        if "label" in batch[0].keys():
            labels = [item["label"] for item in batch]
            item["label"] = labels
        return item
    
    
class InferenceOCRDataset(Dataset):
    def __init__(self, images: list):
        super().__init__()
        transform_list = [
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Resize((32, 80))
        ]
        self.images = images
        self.nSamples = len(self.images)
        self.transform = transforms.Compose(transform_list)
        self.collate_fn = SynthCollator()

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), "index range error"
        img = self.images[index]
        img = F.to_pil_image(img)
        if self.transform is not None:
            img = self.transform(img)
        item = {"img": img, "idx": index}
        return item


class SynthDataset(Dataset):
    def __init__(self, path: str, imgdir: str):
        super(SynthDataset, self).__init__()
        self.path = os.path.join(path, imgdir)
        self.images = os.listdir(self.path)
        self.nSamples = len(self.images)
        f = lambda x: os.path.join(self.path, x)
        self.imagepaths = list(map(f, self.images))
        transform_list = [
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
        self.transform = transforms.Compose(transform_list)
        self.collate_fn = SynthCollator()

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), "index range error"
        imagepath = self.imagepaths[index]
        imagefile = os.path.basename(imagepath)
        img = Image.open(imagepath)
        if self.transform is not None:
            img = self.transform(img)
        #             img = TF.autocontrast(img)
        item = {"img": img, "idx": index}
        item["label"] = imagefile.split("_")[0]
        return item