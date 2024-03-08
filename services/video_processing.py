from typing import List

import cv2
import numpy as np

from .face_detection import FaceDetection, preprocess_image

CAMERA_INDEX: int = 0

BBOX_PURPLE = (255, 255, 0)
BBOX_RED = (255, 0, 0)
BBOX_GREEN = (0, 255, 0)
BBOX_BLUE = (0, 0, 255)


class VideoProcessing:
    webcam: cv2.VideoCapture
    faces: List[np.ndarray]

    def __init__(self, camera_index: int = None):
        if camera_index is not None:
            self.webcam = cv2.VideoCapture(camera_index)
        self.webcam = cv2.VideoCapture(CAMERA_INDEX)

    def stream(self):
        detector = FaceDetection()
        detector.load_model()

        while self.webcam.isOpened():
            ret, frame = self.webcam.read()
            image = preprocess_image(frame)

            boxes, _, _ = detector.detect(frame, image)

            for bbox in range(boxes.shape[0]):
                box = boxes[bbox, :]
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), BBOX_GREEN, 4)

            cv2.imshow('Face Detection Market', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.webcam.release()
        cv2.destroyAllWindows()

    def pipeline_to_streamlit(self, frame_window):
        detector = FaceDetection()
        detector.load_model()

        while self.webcam.isOpened():
            ret, frame = self.webcam.read()
            if ret:
                image = preprocess_image(frame)

                boxes, _, _ = detector.detect(frame, image)

                for bbox in range(boxes.shape[0]):
                    box = boxes[bbox, :]
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), BBOX_GREEN, 4)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_window.image(frame)
