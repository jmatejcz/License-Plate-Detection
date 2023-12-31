import os
import numpy as np
import torch
from PIL import Image
import cv2
import copy
import time
import torchvision
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor,
)
import alpr.ai_utils.transforms as transforms
import alpr.ai_utils as ai_utils
import alpr.utils.image_utils as image_utils
from alpr.plate_segmentation.datasets import InstanceSegmentationDataset

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
def get_fasterrcnn_object_detection(num_classes: int, default_weights: bool):
    """Returns fasterrcnn_resnet50_fpn_v2 model for object detection
    with replaced classificator to match number of classes
    """
    if default_weights:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights=None, weights_backbone="DEFAULT", trainable_backbone_layers=3
        )
    else:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=None)
    num_classes = 2  # 1 class licensese plate + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


class LicensePlatesDetection:
    def __init__(self, model, dataset, batch_size=1, train_split: float = 0.8):
        """Used for training and evaluating models"""
        self.model = model
        train_set, test_set = torch.utils.data.random_split(
            dataset, [train_split, 1 - train_split]
        )
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

    def train(self, num_epochs: int, save_path=None, show_fr: int = 0):
        self.model = self.model.to(self.device)
        self.epochs_losses_train = []
        best_IoU = 0
        start_time = time.time()

        for epoch in range(num_epochs):
            loss_value = self.train_epoch(show_fr=show_fr)
            self.epochs_losses_train.append(loss_value)
            print(f"train_epoch: {epoch}, loss: {loss_value:.4f}")

            IoUs = self.eval(score_threshold=0.9, show_fr=show_fr)
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
                    if targets[0]["boxes"].shape[0] > 0:
                        visual = image_utils.draw_bboxes(
                            inputs[0], targets[0]["boxes"], targets[0]["labels"]
                        )
                        image_utils.show(visual)

            self.optimizer.zero_grad()
            losses_dict = self.model(inputs, targets)
            losses = sum(loss for loss in losses_dict.values())
            loss_value = losses.item()
            losses.backward()
            self.optimizer.step()

        self.lr_scheduler.step()
        self.epochs_losses_train.append(loss_value)

        return loss_value

    def eval(self, show_fr: int = 0, score_threshold: float = 0.9) -> list:
        self.model.eval()
        self.model = self.model.to(self.device)
        iou_scores = []
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.dataloaders["test"]):
                inputs = list(image.to(self.device) for image in inputs)
                targets = [
                    {k: v.to(self.device) for k, v in t.items()} for t in targets
                ]
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
                            if outs[0]["boxes"].shape[0] > 0:
                                visual = image_utils.draw_bboxes(
                                    inputs[0],
                                    outs[0]["boxes"].unsqueeze(0),
                                    color=(0, 0, 255),
                                )
                                image_utils.show(visual)

                    if len(filtered_outputs) > 0:
                        if targets[0]["boxes"].shape[0] < 1:
                            # no bboxes in target
                            iou_scores = [0.0]
                        else:

                            iou_scores.append(
                                # TODO targets [?] zrobic zeby dla bathc wiecej niz 1 tez dzialalo
                                ai_utils.metrics.get_IoU(
                                    outs[0]["boxes"],
                                    targets[0]["boxes"].cpu().numpy()[0],
                                )
                            )
                    else:
                        iou_scores = [0.0]

        return iou_scores


class PlateCropper:
    def __init__(self, model, images: list) -> None:
        """Can crop images to only License Plate, using trained model"""
        self.images = images
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = InstanceSegmentationDataset(
            images=self.images, transforms=transforms.get_transform(train=False)
        )
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
        )

    def crop_plates(
        self,
        score_threshold: float = 0.9,
        crop_enlarge: float = 1.0,
    ) -> list:
        """Crop plates and returns new images as only boxes of detected plates or None if no plate was detected

        :param crop_enlarge: cropes larger or smaller area than box, defaults to 1.0
        :type crop_enlarge: float, optional
        """
        self.model.eval()
        self.model = self.model.to(self.device)
        croped_plates = []
        scores = []

        with torch.no_grad():
            for i, inputs in enumerate(self.dataloader):
                inputs = list(image.to(self.device) for image in inputs)
                outputs = self.model(inputs)

                if len(outputs[0]["boxes"]) > 0:
                    if outputs[0]["scores"][0] > score_threshold:
                        best_box = outputs[0]["boxes"][0]
                        if crop_enlarge != 1.0:
                            best_box = image_utils.enlarge_box(
                                inputs[0].shape, best_box, crop_enlarge
                            )
                        box_img = image_utils.get_box_img(inputs[0], best_box) * 255
                        croped_plates.append(
                            box_img.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
                        )
                        scores.append(outputs[0]["scores"][0])
                    else:
                        print(f"No plate detected in {i+1} image")
        return croped_plates, scores

    def load_state_dict(self, path_to_weights: str):
        self.model.load_state_dict(torch.load(path_to_weights))


class AlprSetupPlateCrop(PlateCropper):
    def __init__(
        self,
        model,
        root: str,
        imgs_folder: str,
        idx_start: int = 0,
        idx_end: int = None,
    ) -> None:
        """Extended version of PlateCropper, can save results, tracks time, shows images and results

        :param idx_start: from which file in folder beign loading, defaults to 0
        :type idx_start: int, optional
        :param idx_end: On which file in folder to end,
        if not provided take all, defaults to None
        :type idx_end: int, optional
        """
        self.root = root
        self.path_to_imgs = imgs_folder
        self.fileterd_img_names = []
        self.ordered_images, self.fileterd_img_names = image_utils.load_images(
            root + imgs_folder, idx_start=idx_start, idx_end=idx_end
        )
        super().__init__(model=model, images=self.ordered_images)

    def crop_plates(
        self,
        path_to_save: str,
        score_threshold: float = 0.9,
        show_fr: int = 0,
        remove_empty: bool = False,
        crop_enlarge: float = 1.0,
    ):
        """Crop plates and save new images as only boxes of detected plates

        :param show_fr: every <number> image will be shown, defaults to 0
        :type show_fr: int, optional
        :param remove_empty: removes empty images from oryginal directory, defaults to False
        :type remove_empty: bool, optional
        :param crop_enlarge: cropes larger or smaller area than box, defaults to 1.0
        :type crop_enlarge: float, optional
        """
        self.model.eval()
        self.model = self.model.to(self.device)
        time_start = time.time()

        with torch.no_grad():
            for i, inputs in enumerate(self.dataloader):
                inputs = list(image.to(self.device) for image in inputs)
                outputs = self.model(inputs)

                if len(outputs[0]["boxes"]) > 0:
                    if show_fr > 0:
                        if i % show_fr == 0:
                            boxes = []
                            for u, box in enumerate(outputs[0]["boxes"]):
                                if outputs[0]["scores"][u] > score_threshold:
                                    boxes.append(box.cpu().numpy())

                                boxes = torch.as_tensor(boxes)
                                visual = image_utils.draw_bboxes(inputs[0], boxes)
                                if isinstance(visual, torch.Tensor):
                                    print(outputs[0]["scores"], boxes)
                                    image_utils.show(visual)

                    if outputs[0]["scores"][0] > score_threshold:
                        best_box = outputs[0]["boxes"][0]
                        if crop_enlarge != 1.0:
                            best_box = image_utils.enlarge_box(
                                inputs[0].shape, best_box, crop_enlarge
                            )
                        path_to_save_file = path_to_save + self.fileterd_img_names[i]
                        box_img = image_utils.get_box_img(inputs[0], best_box)
                        box_img = 255 * np.transpose(box_img.cpu().numpy(), (1, 2, 0))
                        cv2.imwrite(path_to_save_file, box_img)

                elif remove_empty:
                    os.remove(
                        f"{self.root}{self.path_to_imgs}{self.fileterd_img_names[i]}"
                    )

                if i % 100 == 0:
                    time_end = time.time()
                    print(f"images done -> {i}, time_spent: {time_end-time_start}")
        time_end = time.time()
        print(f"time -> {time_end - time_start}")
