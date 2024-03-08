import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort

from .utils import download_model, get_model_url

BASE_PATH = Path(__file__).parent.parent


class UltraFaceModelOptions(Enum):
    VERSION_RFB_320: str = "version-RFB-320.onnx"
    VERSION_RFB_320_INT8: str = "version-RFB-320-int8.onnx"
    VERSION_RFB_640: str = "version-RFB-640.onnx"


DEFAULT_MODEL = UltraFaceModelOptions.VERSION_RFB_320


@dataclass()
class FacialAreaRegion:
    x: int
    y: int
    w: int
    h: int
    left_eye: Tuple[int, int]
    right_eye: Tuple[int, int]
    confidence: float


@dataclass()
class DetectionResult:
    image: np.ndarray
    facial_area_regions: FacialAreaRegion
    confidence: float


def area_of(left_top: np.ndarray, right_bottom: np.ndarray):
    """
    Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0: np.ndarray, boxes1: np.ndarray, eps=1e-5):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def hard_nms(
        box_scores: np.ndarray,
        iou_threshold: int,
        top_k: int = -1,
        candidate_size: int = 200):
    """
    Perform hard non-maximum-suppression to filter out boxes with iou greater
    than threshold
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
        picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    # _, indexes = scores.sort(descending=True)
    indexes = np.argsort(scores)
    # indexes = indexes[:candidate_size]
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        # current = indexes[0]
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        # indexes = indexes[1:]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]


def filter_boxes_with_threshold(
        width: int,
        height: int,
        confidences: np.ndarray,
        boxes: np.ndarray,
        prob_threshold: float,
        iou_threshold=0.5,
        top_k=-1):
    """
    Select boxes that contain human faces
    Args:
        width: original image width
        height: original image height
        confidences (N, 2): confidence array
        boxes (N, 4): boxes array in corner-form
        iou_threshold: intersection over union threshold.
        prob_threshold: probability threshold for detection.
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
        boxes (k, 4): an array of boxes kept
        labels (k): an array of labels for each boxes kept
        probs (k): an array of probabilities for each boxes being in corresponding labels
    """
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(box_probs,
                             iou_threshold=iou_threshold,
                             top_k=top_k,
                             )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


def preprocess_image(image: np.ndarray) -> np.ndarray:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (320, 240))
    image_mean = np.array([127, 127, 127])
    image = (image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    return image


class FaceDetection:
    ort_session: ort.InferenceSession
    model_dir: str
    default_model: UltraFaceModelOptions = DEFAULT_MODEL
    model_path: str
    input_name: str

    def __init__(self):
        self.model_dir = os.path.join(BASE_PATH, "models")

    def load_model(self, model_name: Union[UltraFaceModelOptions, None] = None) -> None:
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        if model_name is None:
            if not os.path.isfile(os.path.join(self.model_dir, self.default_model.value)):
                download_model(get_model_url(self.default_model.value),
                               os.path.join(self.model_dir, self.default_model.value))
            self.model_path = os.path.join(self.model_dir, self.default_model.value)
        else:
            if not os.path.isfile(os.path.join(self.model_dir, model_name.value)):
                download_model(get_model_url(model_name.value), os.path.join(self.model_dir, model_name.value))
            self.model_path = os.path.join(self.model_dir, model_name.value)

        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3

        self.ort_session = ort.InferenceSession(self.model_path, sess_options=session_options)
        self.input_name = self.ort_session.get_inputs()[0].name

    def detect(self, frame, image, threshold: float = 0.7):
        confidences, boxes = self.ort_session.run(None, {
            self.input_name: image
        })
        return filter_boxes_with_threshold(
            frame.shape[1],
            frame.shape[0],
            confidences,
            boxes,
            threshold
        )
