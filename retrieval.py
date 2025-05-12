import cv2
import numpy as np


def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union_area = w1 * h1 + w2 * h2 - inter_area

    return inter_area / union_area if union_area != 0 else 0


def non_max_suppression(boxes, iou_threshold=0.3):
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda x: x[0])
    selected = []

    while boxes:
        current = boxes.pop(0)
        boxes = [box for box in boxes if calculate_iou(current, box) <= iou_threshold]
        selected.append(current)

    return selected


def enhance_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def predict_image(img: np.ndarray, query: np.ndarray) -> list:
    img = enhance_contrast(img)
    query = enhance_contrast(query)

    orb = cv2.ORB_create(nfeatures=4000)
    kp1, des1 = orb.detectAndCompute(query, None)
    kp2, des2 = orb.detectAndCompute(img, None)

    boxes = []

    if des1 is not None and des2 is not None:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good_matches) >= 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is not None and mask.sum() > 12:
                h, w = query.shape[:2]
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)

                x_coords = np.clip(dst[:, 0, 0], 0, img.shape[1])
                y_coords = np.clip(dst[:, 0, 1], 0, img.shape[0])

                x_min = np.min(x_coords)
                y_min = np.min(y_coords)
                width = np.max(x_coords) - x_min
                height = np.max(y_coords) - y_min

                if width > 15 and height > 15:
                    boxes.append((
                        x_min / img.shape[1],
                        y_min / img.shape[0],
                        width / img.shape[1],
                        height / img.shape[0]
                    ))

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    query_gray = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)

    best_val = 0
    best_box = None
    scales = np.linspace(0.6, 1.4, 9)
    for scale in scales:
        resized = cv2.resize(query_gray, None, fx=scale, fy=scale)
        if resized.shape[0] > img_gray.shape[0] or resized.shape[1] > img_gray.shape[1]:
            continue

        res = cv2.matchTemplate(img_gray, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > best_val and max_val > 0.65:
            best_val = max_val
            x, y = max_loc
            best_box = (
                x / img.shape[1],
                y / img.shape[0],
                resized.shape[1] / img.shape[1],
                resized.shape[0] / img.shape[0]
            )

    if best_box:
        boxes.append(best_box)

    boxes = non_max_suppression(boxes)
    return boxes