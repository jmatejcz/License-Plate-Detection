from __future__ import print_function
import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_images_by_frames(img_path: str, frames: list):
    images = []
    for frame in frames:
        file = img_path + f"buspas_2_lane_3_1_{frame}.jpg"
        images.append(cv2.imread(file, cv2.IMREAD_COLOR))
    return images


def get_keypoints_and_matches(
    images: list,
    max_features: int = 500,
    good_matches_percent: float = 0.5,
    edge_threshold: int = 1,
    matching_with_first: bool = True,
    show_matches: bool = True,
):
    """Find keypoint in every img than find matches beetwen images

    :param matching_with_first: If true all images are matching its keypoints. If False every image matches with the next one, defaults to True
    :type matching_with_first: bool, optional

    :return: _description_
    :rtype: _type_
    """
    
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_SL2)
    orb = cv2.ORB_create(max_features, 1, edgeThreshold=edge_threshold, patchSize=20)

    # Detect keypoints
    keypoints = []
    descriptors = []
    matches = []
    for img in images:
        im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _keypoints, _descriptors = orb.detectAndCompute(im_gray, None)
        keypoints.append(_keypoints)
        descriptors.append(_descriptors)

    # Match features
    for i in range(1, len(descriptors)):
        if matching_with_first:
            one_img_matches = list(matcher.match(descriptors[i], descriptors[0], None))
        else:
            one_img_matches = list(matcher.match(descriptors[i], descriptors[i - 1], None))
        # Sort matches by score
        one_img_matches.sort(key=lambda x:x.distance)
        numGoodMatches = int(len(one_img_matches) * good_matches_percent)
        one_img_matches = one_img_matches[:numGoodMatches]
        matches.append(one_img_matches)

        # Draw top matches from the current and next image
        if show_matches:
            if matching_with_first:
                print(
                    f"number of matches between img_{i} and img_{0}-> {len(one_img_matches)}"
                )
                imMatches = cv2.drawMatches(
                    images[i],
                    keypoints[i],
                    images[0],
                    keypoints[0],
                    one_img_matches,
                    None,
                )
                plt.imshow(imMatches)
                plt.show()

            else:
                print(
                    f"number of matches between img_{i} and img_{i-1}-> {len(one_img_matches)}"
                )
                imMatches = cv2.drawMatches(
                    images[i],
                    keypoints[i],
                    images[i - 1],
                    keypoints[i - 1],
                    one_img_matches,
                    None,
                )
                plt.imshow(imMatches)
                plt.show()

    return keypoints, matches


def get_homography_matrix(keypoints, matches):
    homography_matrix = np.zeros((3, 3))
    # Extract location of matches
    for i in range(len(keypoints) - 1):
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)
        if len(matches[i]) < 1:
            continue
        else:
            for u, match in enumerate(matches[i]):
                points1[u] = keypoints[i][match.queryIdx].pt
                points2[u] = keypoints[i + 1][match.trainIdx].pt
            # Find homography between current and next image
            if points1.shape[0] >= 4 or points2.shape[0] >= 4:
                homography_matrix = np.zeros((3, 3))
                h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
                homography_matrix += h
            else:
                print(
                    "you need at least 4 coresponing points to calculate matrix. Probably you have too few matches"
                )

    mean_h_mat = homography_matrix / len(keypoints)

    return mean_h_mat


def transform_image(
    image,
    homography_matrix,
    show_image: bool = False,
    save_path=None,
):
    (h, w) = image.shape[:2]
    aligned = cv2.warpPerspective(image, homography_matrix, (w, h))
    if show_image:
        plt.imshow(aligned)
        plt.show()
    if save_path:
        cv2.imwrite(save_path, aligned)
    return aligned
