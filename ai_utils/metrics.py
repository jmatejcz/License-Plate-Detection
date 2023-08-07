def get_IoU(box1: list, box2: list) -> float:
    """Calculates Intersection over Union bettwen 2 boxes

    :return: 0 - 1 , where 1 means fully alligned boxes
    :rtype: float
    """
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

