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

def save_frames_from_video_moviepy(video_path, output_dir, interval=None):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the video
    video_clip = VideoFileClip(video_path)

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

    # Close the video clip
    video_clip.reader.close()
    
def save_frames_from_video_opencv(video_folder, video_filename, save_path):
    frames = []
    vid_capture= cv2.VideoCapture(video_folder+video_filename)
    count = 0
    while(vid_capture.isOpened()):
        ret, frame = vid_capture.read()
        if ret == True:
            print("xd")
            cv2.imwrite(f"{save_path}{video_filename}_{count}.jpg", frame)
            frames.append(frame)
        else:
            cap.release()
            cv2.destroyAllWindows()
        count += 1
    return frames

def save_frames(video_folder, video_filename, save_path):
    frames = []
    print(video_folder + video_filename)
    vid_capture= cv2.VideoCapture(video_folder+video_filename)
    count = 0
    while(vid_capture.isOpened()):
        ret, frame = vid_capture.read()
        print(ret)
        if ret == True:
            print("xd")
            cv2.imwrite(f"{save_path}{video_filename}_{count}.jpg", frame)
            frames.append(frame)
        else:
            cap.release()
            cv2.destroyAllWindows()
        count += 1
    return frames


def scale_to_multiple_of_16(img):
    pass

def load_images(folder_path: str, idx_start: int, idx_end: int)-> tuple:
    """Loads certain amount of sorted images from folder and their filenames

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
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()

def show_multiple_imgs(images, titles = None, inch_size = (40, 40)):
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

def draw_bboxes(image, boxes, labels=None, scores=None, color = (255, 0, 0)):
    image = torch.as_tensor(image * 255, dtype=torch.uint8)
    if boxes.shape[0] == 0:
        return
    else:
        visual = draw_bounding_boxes(image, boxes, colors=color, width=5)
    return visual

def get_box_img(image: torch.Tensor, boxes: torch.Tensor):
    croped_img = image[:, int(boxes[1]) : int(boxes[3]), int(boxes[0]) : int(boxes[2])]
#     croped_img = F.to_pil_image(croped_img)
    return croped_img

def enlarge_box(img_size: list, box: list, magnify: float):
    box_x = (box[2] - box[0]) / 2
    box_y = (box[3] - box[1]) / 2
    if box[0] - box_x * (magnify - 1.0) < 0:
        box[0] = 0.0
    else:
        box[0] -= box_x * (magnify - 1.0)
    if box[2] + box_x * (magnify - 1.0) > img_size[2]:
        box[2] = img_size[2]
    else:
        box[2] += box_x * (magnify - 1.0)

    if box[1] - box_y * (magnify - 1.0) < 0:
        box[1] = 0.0
    else:
        box[1] -= box_y * (magnify - 1.0)
    if box[3] + box_y * (magnify - 1.0) > img_size[1]:
        box[3] = img_size[1]
    else:
        box[3] += box_y * (magnify - 1.0)
    return box
