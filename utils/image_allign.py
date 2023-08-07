import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_keypoints_and_matches(
    images: list,
    max_features: int = 500,
    good_matches_percent: float = 0.5,
    edge_threshold: int = 1,
    matching_with_first: bool = True,
    show_matches: bool = True,
) -> tuple:
    """Find keypoint in every img than find matches beetwen images

    :param matching_with_first: If true all images are matching its keypoints. If False every image matches with the next one, defaults to True
    :type matching_with_first: bool, optional

    :return: list of keypoints and list of matches
    :rtype: tuple(list, list)
    """

    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_SL2)
    orb = cv2.ORB_create(max_features, 1, edgeThreshold=edge_threshold, patchSize=20)

    # Detect keypoints
    keypoints = []
    descriptors = []
    matches = []
    for img in images:
        im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        one_img_keypoints, one_img_descriptors = orb.detectAndCompute(im_gray, None)
        keypoints.append(one_img_keypoints)
        descriptors.append(one_img_descriptors)
    # Match features
    for i in range(1, len(descriptors)):
        if matching_with_first:
            one_img_matches = list(matcher.match(descriptors[i], descriptors[0], None))
        else:
            one_img_matches = list(
                matcher.match(descriptors[i], descriptors[i - 1], None)
            )
        # Sort matches by score
        one_img_matches.sort(key=lambda x: x.distance)
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


def get_homography_matrix(keypoints: list, matches: list) -> np.ndarray:
    """Homography is the mapping between two planar projections of an image

    :param keypoints: _description_
    :type keypoints: list
    :param matches: _description_
    :type matches: list
    :return: _description_
    :rtype: np.ndarray
    """
    homography_matrix = np.zeros((3, 3))
    # Extract location of matches
    for i in range(len(keypoints) - 1):
        if len(matches[i]) < 1:
            continue
        else:
            points1 = np.zeros((len(matches[i]), 2), dtype=np.float32)
            points2 = np.zeros((len(matches[i]), 2), dtype=np.float32)
            for u, match in enumerate(matches[i]):
                points1[u] = keypoints[i][match.trainIdx].pt
                points2[u] = keypoints[i + 1][match.queryIdx].pt
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
    homography_matrix: np.ndarray,
    show_image: bool = False,
    save_path=None,
):
    """ransforms image using homography matrix

    :param image: _description_
    :type image: _type_
    :param homography_matrix: _description_
    :type homography_matrix: np.ndarray
    :param show_image: _description_, defaults to False
    :type show_image: bool, optional
    :param save_path: _description_, defaults to None
    :type save_path: _type_, optional
    :return: _description_
    :rtype: _type_
    """
    (h, w) = image.shape[:2]
    aligned = cv2.warpPerspective(image, homography_matrix, (w, h))
    if show_image:
        plt.imshow(aligned)
        plt.title("aligned")
        plt.show()

        plt.imshow(image)
        plt.title("original")
        plt.show()

    if save_path:
        cv2.imwrite(save_path, aligned)
    return aligned
