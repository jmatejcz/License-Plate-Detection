import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
import cv2
import os
from PIL import Image
import torchvision.transforms.functional as F
import numpy as np
import math
import imageio
import moviepy
from numpy.typing import ArrayLike

def save_frames_from_video_moviepy(video_path: str, output_dir: str, interval: float=None) -> None:
    """Saves frames from given video, using moviepy

    :param interval: time between frames to cut, if None every frame will be returned, defaults to None
    :type interval: float, optional
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_clip = moviepy.VideoFileClip(video_path)

    # Get the total duration of the video inseconds
    total_duration = video_clip.duration
    if not interval:
        interval  = 1 / video_clip.fps
    
    # Save frames at specified intervals
    for t in range(int(total_duration/interval)):
        time_point = t * interval
        frame = video_clip.get_frame(time_point)
        frame_path = os.path.join(output_dir, f"frame_{t}.png")
        imageio.imwrite(frame_path, frame)

    video_clip.reader.close()
    
def save_frames_from_video_opencv(video_folder: str, video_filename: str, save_path: str) -> list:
    """Save frames from video using opencv

    :param video_folder: _description_
    :type video_folder: str
    :param video_filename: _description_
    :type video_filename: str
    :param save_path: _description_
    :type save_path: str
    :return: _description_
    :rtype: list
    """
    frames = []
    vid_capture = cv2.VideoCapture(video_folder+video_filename)
    count = 0
    while(vid_capture.isOpened()):
        ret, frame = vid_capture.read()
        if ret == True:
            print("xd")
            cv2.imwrite(f"{save_path}{video_filename}_{count}.jpg", frame)
            frames.append(frame)
        else:
            vid_capture.release()
            cv2.destroyAllWindows()
        count += 1
    return frames


def load_images(folder_path: str, idx_start: int, idx_end: int)-> tuple:
    """Loads images as PIL Images

    :param folder_path: _description_
    :type folder_path: str
    :param idx_start: image in folder from which to start
    :type idx_start: int
    :param idx_end: image in folder on which to end
    :type idx_end: int
    :return: return images and their filenames
    :rtype: tuple <list, list>
    """
    images = []
    filenames = []
    for file in list(sorted(os.listdir(folder_path))[idx_start:idx_end]):
        try:
            image = Image.open(f"{folder_path}{file}").convert(
                "RGB"
            )
            images.append(image)
            filenames.append(file)
        except:
            print(f"could not open {folder_path}{file}")
    return images, filenames

def show(imgs):
    """Shows image as plt figure

    :param imgs: Images as tensors
    """
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()

def show_multiple_imgs(images, titles = None, inch_size = (40, 40)) -> None:
    """Creates plt figure to show multiple images

    :param images: _description_
    :type images: _type_
    :param titles: _description_, defaults to None
    :type titles: _type_, optional
    :param inch_size: _description_, defaults to (40, 40)
    :type inch_size: tuple, optional
    """
    cols = 5
    rows = math.ceil(len(images)/5)
    fig, axs = plt.subplots(rows, cols)
    fig.set_size_inches(*inch_size)
    for i, img in enumerate(images):
        if titles:
            axs[i//cols, i%cols].title.set_text(titles[i])
        else:    
            axs[i//cols, i%cols].title.set_text(f"{i+1}")
        axs[i//cols, i%cols].imshow(img)

def draw_bboxes(image: torch.Tensor | np.ndarray, boxes: torch.Tensor | np.ndarray, color: tuple = (255, 0, 0)) -> torch.Tensor:
    """Draws bounding boxes on image

    :param image: tensor or numpy ndarray with values 0-255
    :type image: torch.Tensor | np.ndarray
    :param boxes: tensor or numpy ndarray with values 0-255
    :type boxes: torch.Tensor | np.ndarray
    :param color: RGB color, defaults to (255, 0, 0)
    :type color: tuple, optional
    :return: image with bbox 
    :rtype: torch.Tensor
    """
    image = torch.as_tensor(image, dtype=torch.uint8)
    boxes = torch.as_tensor(boxes, dtype=torch.uint8)
    if image.shape[0] != 3 or len(image.shape) != 3:
        raise ValueError(f"Only RGB images with 3 channels are supported, you provided iamge with shape {image.shape}")
    if len(boxes.shape) != 2 or boxes.shape[1] != 4:
        raise ValueError(f"Boxes are supported in shape ([N x 4]), you provided {boxes.shape}")
    if boxes.shape[0] == 0:
        return image
    else:
        visual = draw_bounding_boxes(image, boxes, colors=color, width=5)
    return visual

def get_box_img(image: torch.Tensor | np.ndarray, boxes: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """crop image to only bouding box area

    :param image: _description_
    :type image: torch.Tensor | np.ndarray
    :param boxes: _description_
    :type boxes: torch.Tensor | np.ndarray
    :return: _description_
    :rtype: torch.Tensor | np.ndarray
    """
    croped_img = image[:, int(boxes[1]) : int(boxes[3]), int(boxes[0]) : int(boxes[2])]
    return croped_img

def enlarge_box(img_shape: ArrayLike, box: ArrayLike, magnify: float) -> ArrayLike:
    """Enlarge box within given image

    :param img_shape: (C x H x W)
    :type img_shape: ArrayLike
    :param box: [x0, y0, x1, y1]
    :type box: ArrayLike
    :param magnify: box enlargment, every edge is magnified,
    so area of box will be magnify^2 enlarged 
    :type magnify: float
    :return: enlarged box
    :rtype: ArrayLike
    """
    box_x = (box[2] - box[0]) / 2
    box_y = (box[3] - box[1]) / 2
    if box[0] - box_x * (magnify - 1.0) < 0:
        box[0] = 0.0
    else:
        box[0] -= box_x * (magnify - 1.0)
    if box[2] + box_x * (magnify - 1.0) > img_shape[2]:
        box[2] = img_shape[2]
    else:
        box[2] += box_x * (magnify - 1.0)

    if box[1] - box_y * (magnify - 1.0) < 0:
        box[1] = 0.0
    else:
        box[1] -= box_y * (magnify - 1.0)
    if box[3] + box_y * (magnify - 1.0) > img_shape[1]:
        box[3] = img_shape[1]
    else:
        box[3] += box_y * (magnify - 1.0)
    return box
