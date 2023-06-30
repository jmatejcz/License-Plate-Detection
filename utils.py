import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
import cv2
import torchvision.transforms.functional as F
import numpy as np

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
    return visual


def save_boxes_img(path_to_save: str, image: torch.Tensor, boxes: torch.Tensor):
    croped_img = image[:, int(boxes[1]) : int(boxes[3]), int(boxes[0]) : int(boxes[2])]
    croped_img = F.to_pil_image(croped_img)
    cv2.imwrite(path_to_save, np.asarray(croped_img))


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