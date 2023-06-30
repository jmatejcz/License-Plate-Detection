import os
import numpy as np
import torch
from PIL import Image
import json
import copy
import time
import torchvision
from torchvision.utils import draw_bounding_boxes
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision import transforms as T
import torchvision.transforms.functional as F
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union


#     ⠀⠀⠀⠀⠀⠀⠀⠀⣠⣤⣤⣤⣤⣤⣶⣦⣤⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⣿⡿⠛⠉⠙⠛⠛⠛⠛⠻⢿⣿⣷⣤⡀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⣼⣿⠋⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⠈⢻⣿⣿⡄⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⣸⣿⡏⠀⠀⠀⣠⣶⣾⣿⣿⣿⠿⠿⠿⢿⣿⣿⣿⣄⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⣿⣿⠁⠀⠀⢰⣿⣿⣯⠁⠀⠀⠀⠀⠀⠀⠀⠈⠙⢿⣷⡄⠀
# ⠀⠀⣀⣤⣴⣶⣶⣿⡟⠀⠀⠀⢸⣿⣿⣿⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣷⠀
# ⠀⢰⣿⡟⠋⠉⣹⣿⡇⠀⠀⠀⠘⣿⣿⣿⣿⣷⣦⣤⣤⣤⣶⣶⣶⣶⣿⣿
# ⠀⢸⣿⡇⠀⠀⣿⣿⡇⠀⠀⠀⠀⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿
# ⠀⣸⣿⡇⠀⠀⣿⣿⡇⠀⠀⠀⠀⠀⠉⠻⠿⣿⣿⣿⣿⡿⠿⠿⠛⢻⣿⡇⠀⠀
# ⠀⣿⣿⠁⠀⠀⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣧⠀⠀
# ⠀⣿⣿⠀⠀⠀⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⠀⠀
# ⠀⣿⣿⠀⠀⠀⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⠀⠀
# ⠀⢿⣿⡆⠀⠀⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⡇⠀⠀
# ⠀⠸⣿⣧⡀⠀⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⠃⠀⠀
# ⠀⠀⠛⢿⣿⣿⣿⣿⣇⠀⠀⠀⠀⠀⣰⣿⣿⣷⣶⣶⣶⣶⠶⢠⣿⣿⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⣿⣿⠀⠀⠀⠀⠀⣿⣿⡇⠀⣽⣿⡏⠁⠀⠀⢸⣿⡇⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⣿⣿⠀⠀⠀⠀⠀⣿⣿⡇⠀⢹⣿⡆⠀⠀⠀⣸⣿⠇⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⢿⣿⣦⣄⣀⣠⣴⣿⣿⠁⠀⠈⠻⣿⣿⣿⣿⡿⠏⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠈⠛⠻⠿⠿⠿⠿⠋⠁⠀⠀
def get_model_instance_segmentation(num_classes, default: bool):
    if default:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights=None, weights_backbone="DEFAULT", trainable_backbone_layers=3
        )
    else:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=None)
    num_classes = 2  # 1 class (person) + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


def draw_bboxes(image, boxes, labels=None):
    image = torch.as_tensor(image * 255, dtype=torch.uint8)
    if boxes.shape[1] == 0:
        return
    else:
        visual = draw_bounding_boxes(image, boxes, colors=(255, 0, 0), width=2)
        show(visual)


def save_boxes_img(path_to_save: str, image: torch.Tensor, boxes: torch.Tensor):
    croped_img = image[:, int(boxes[1]) : int(boxes[3]), int(boxes[0]) : int(boxes[2])]
    croped_img = F.to_pil_image(croped_img)
    cv2.imwrite(path_to_save, np.asarray(croped_img))


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(
        self, image: torch.Tensor, target: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if target is not None:
                _, _, width = F.get_dimensions(image)
                target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
                if "masks" in target:
                    target["masks"] = target["masks"].flip(-1)
                if "keypoints" in target:
                    keypoints = target["keypoints"]
                    keypoints = _flip_coco_person_keypoints(keypoints, width)
                    target["keypoints"] = keypoints
        return image, target


class PILToTensor(torch.nn.Module):
    def forward(
        self, image: torch.Tensor, target: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        image = F.pil_to_tensor(image)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class ConvertImageDtype(torch.nn.Module):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.dtype = dtype

    def forward(
        self, image: torch.Tensor, target: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        image = F.convert_image_dtype(image, self.dtype)
        return image, target


def get_transform(train: bool):
    transforms = []
    if train:
        transforms.append(PILToTensor())
        transforms.append(ConvertImageDtype(torch.float))
        return Compose(transforms)
    else:
        transforms.append(T.PILToTensor())
        transforms.append(T.ConvertImageDtype(torch.float))
        return T.Compose(transforms)


def get_IoU(box1: list, box2: list) -> float:
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[0] + box1[2], box2[0] + box2[2])
    y_bottom = min(box1[1] + box1[3], box2[1] + box2[3])
    # If the intersection is non-existent (boxes don't overlap)
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    # Calculate the areas of the bounding boxes
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area

    return iou


class InstanceSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, root, images_path, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(self.root + images_path)))[:10000]
        self.ordered_images = []
        
        for file_img_path in self.imgs:
            try:
                image = Image.open(f"{root}{images_path}{file_img_path}").convert("RGB")
                self.ordered_images.append(image)
            except:
                print(f"could not open {file_img_path}")

    def __getitem__(self, idx):
        if self.transforms is not None:
            img = self.transforms(self.ordered_images[idx])

        return img

    def __len__(self):
        return len(self.imgs)


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

        targets = []
        ann_idx = 0
        img_idx = 0
        for ann in self.labels_dict["annotations"]:
            boxes = []
            ann_id = ann["id"] + ann_idx
            img_id = ann["image_id"] + img_idx
            if ann_id < img_id:
                # image with no annotations
                boxes = [0, 0, 0, 0]
                target = {}
                target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
                target["labels"] = torch.zeros([0], dtype=torch.int64)
                target["image_id"] = torch.tensor([img_idx])
                target["area"] = torch.zeros([0], dtype=torch.float32)
                target["iscrowd"] = torch.zeros([0], dtype=torch.int64)
                ann_idx += 1
            elif ann_id > img_id:
                # image with more than 1 annotation
                # ignore?
                print(f"ignoring annotation -> {ann_idx+ann_id}")
                img_idx += 1
                continue

            else:
                # in coco format first 2 values are coords of left up point
                # second 2 values and width and height of box
                for j in range(0, len(ann["bbox"]), 4):
                    x1, y1 = float(ann["bbox"][j]), float(ann["bbox"][j + 1])
                    width, height = float(ann["bbox"][j + 2]), float(ann["bbox"][j + 3])

                    x2, y2 = x1 + width, y1 + height
                    boxes.append([x1, y1, x2, y2])

                target = {}
                target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
                # labels starts from 1, 0 is background
                target["labels"] = torch.as_tensor(
                    [ann["category_id"] + 1], dtype=torch.int64
                )
                target["image_id"] = torch.tensor([ann["image_id"]])
                target["area"] = torch.as_tensor([ann["area"]], dtype=torch.float32)
                target["iscrowd"] = torch.as_tensor([ann["iscrowd"]], dtype=torch.int64)

            targets.append(target)

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


class LicensePlatesDetection:
    """Used for preparing data and training / eval models"""

    def __init__(
        self, model, train_set, test_set, batch_size, model_name=None, train_split=0.8
    ):
        self.model = model
        train_dataloader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch: tuple(zip(*batch)),
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_set,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda batch: tuple(zip(*batch)),
        )
        self.dataloaders = {"train": train_dataloader, "test": test_dataloader}
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=3, gamma=0.1
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_state_dict(self, path_to_weights: str):
        self.model.load_state_dict(torch.load(path_to_weights))

    def train(self, num_epochs: int, save_path=None):
        self.model = self.model.to(self.device)
        self.epochs_losses_train = []
        best_IoU = 0
        start_time = time.time()

        for epoch in range(num_epochs):
            loss_value = self.train_epoch()
            self.epochs_losses_train.append(loss_value)
            print(f"train_epoch: {epoch}, loss: {loss_value:.4f}")

            IoUs = self.eval(score_threshold=0.7)
            mean_IoU = np.mean(IoUs)
            print(f"mean_IoU: {mean_IoU}")
            if save_path:
                if best_IoU < mean_IoU:
                    best_IoU = mean_IoU
                    best_model = copy.deepcopy(self.model.state_dict())
                    torch.save(best_model, f"{save_path}")

        end_time = time.time()
        self.train_time = end_time - start_time
        print(f"train_time: {self.train_time}")

    def train_epoch(self, show_fr: int = 0):
        self.model.train()
        for i, (inputs, targets) in enumerate(self.dataloaders["train"]):
            inputs = list(image.to(self.device) for image in inputs)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            if show_fr > 0:
                if i % show_fr == 0:
                    draw_bboxes(inputs[0], targets[0]["boxes"], targets[0]["labels"])

            self.optimizer.zero_grad()
            losses_dict = self.model(inputs, targets)
            losses = sum(loss for loss in losses_dict.values())
            loss_value = losses.item()
            losses.backward()
            self.optimizer.step()

        self.lr_scheduler.step()
        self.epochs_losses_train.append(loss_value)

        return loss_value

    def eval(
        self, show_fr: int = 0, score_threshold: float = 0.6, save_boxes: bool = False
    ) -> list:
        self.model.eval()
        self.model = self.model.to(self.device)
        iou_scores = []
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.dataloaders["test"]):
                
                inputs = list(image.to(self.device) for image in inputs)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                outputs = self.model(inputs)
                filtered_outputs = []
                for o in outputs:
                    o = {k: v.detach().cpu() for k, v in o.items()}
                    for j, score in enumerate(o["scores"]):
                        if score > score_threshold:
                            filtered_outputs.append([{k: v[j] for k, v in o.items()}])
            
                for j, outs in enumerate(filtered_outputs):
                    if show_fr > 0:
                        if i % show_fr == 0:
                            draw_bboxes(inputs[0], outs[0]["boxes"].unsqueeze(0))

                    if save_boxes:
                        path = f"/workspace/alpr/croped_plates/{i}_{j}.jpg"
                        save_boxes_img(path, inputs[0], outs[0]["boxes"])

                    if len(filtered_outputs) > 0:
                        iou_scores.append(
                            #TODO targets [?] zrobic zeby dla bathc wiecej niz 1 tez dzialalo
                            get_IoU(outs[0]["boxes"], targets[0]["boxes"].cpu().numpy()[0])
                        )
                    else:
                        iou_scores = [0.0]

        return iou_scores


class AlprSetupTraining(LicensePlatesDetection):
    def __init__(
        self,
        model,
        dataset_train,
        dataset_test,
        model_name=None,
        batch_size=1,
        train_split=0.8,
    ):
        self.model_name = model_name
        train_split = int(train_split * len(dataset_train))
        indices_train = torch.randperm(train_split).tolist()
        indices_test = torch.randperm(len(dataset_test) - train_split).tolist()
        train_set = torch.utils.data.Subset(dataset_train, indices_train)
        test_set = torch.utils.data.Subset(dataset_test, indices_test)
        super().__init__(
            model=model,
            train_set=train_set,
            test_set=test_set,
            batch_size=batch_size,
            train_split=0.8,
        )


class AlprSetupPlateCrop:
    "automatic license plate recognition"

    def __init__(self, model, root, path_to_imgs) -> None:
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = InstanceSegmentationDataset(
            root=root, images_path=path_to_imgs, transforms=get_transform(False)
        )
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
        )

    def crop_plates(self, path_to_save: str, score_threshold: float = 0.7, show_fr: int = 0):
        self.model.eval()
        self.model = self.model.to(self.device)
        time_start = time.time()
        with torch.no_grad():
            for i, inputs in enumerate(self.dataloader):
                inputs = list(image.to(self.device) for image in inputs)
                outputs = self.model(inputs)
                            
                if len(outputs[0]["boxes"]) > 0:
                    best_box = outputs[0]['boxes'][0]
                    path_to_save_file = path_to_save + f'{i+50000}.jpg'
                    save_boxes_img(path_to_save_file, inputs[0], best_box)
            
                if i%100 == 0:
                    time_end = time.time()
                    print(f"images done -> {i}, time_spent: {time_end-time_start}")
        
        print(f"time -> {time_end - time_start}")

    def load_state_dict(self, path_to_weights: str):
        self.model.load_state_dict(torch.load(path_to_weights))

if __name__ == "__main__":
    num_classes = 2
    root = "/workspace/alpr/"
    model = get_model_instance_segmentation(num_classes, default=True)
    cropper = AlprSetupPlateCrop(root=root, model=model, path_to_imgs='frames/')
    cropper.load_state_dict(path_to_weights=root+"fasterrcnn_resnet50_fpn_v2.pt")
    cropper.crop_plates(path_to_save=root+'croped_plates/1/')