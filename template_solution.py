import cv2
import numpy as np


def predict_image(img: np.ndarray, query: np.ndarray) -> list:
    list_of_bboxes = [(0, 0, 1, 1), ]
    return list_of_bboxes