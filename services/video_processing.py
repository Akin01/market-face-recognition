from typing import List

import cv2
import numpy as np

from .face_detection import FaceDetection, preprocess_image
from .face_recognition import FaceRecognition

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
        self.webcam = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

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

    def pipeline_to_streamlit(self,
                              frame_window,
                              conf_threshold: float = 0.7,
                              iou_threshold: float = 0.5):
        detector = FaceDetection()
        recognizer = FaceRecognition()

        detector.load_model()

        while self.webcam.isOpened():
            ret, frame = self.webcam.read()

            if ret:
                image = preprocess_image(frame)
                boxes, _, _ = detector.detect(frame, image, conf_threshold, iou_threshold)

                for bbox in range(boxes.shape[0]):
                    box = boxes[bbox, :]
                    extracted_face = frame[box[1]:box[3], box[0]:box[2]]
                    face_embedding = recognizer.generate_embedding(
                        extracted_face
                    )
                    search_result = recognizer.search_face(image_vector=face_embedding)

                    if search_result:
                        face_name = search_result[0].payload.get("fullname")
                        similarity_score = round(search_result[0].score * 100)
                    else:
                        face_name = "unknown"
                        similarity_score = 0

                    # Draw face name to detected face
                    cv2.putText(
                        img=frame,
                        text=face_name,
                        org=(box[0] - 4, box[1] - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=BBOX_BLUE,
                        thickness=1
                    )

                    # Draw similarity search score to detected face
                    cv2.putText(
                        img=frame,
                        text=f"{similarity_score}%",
                        org=(box[2] - 13, box[1] - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(0, 255, 255),
                        thickness=2
                    )

                    # Draw face bounding box
                    cv2.rectangle(
                        frame,
                        (box[0], box[1]),
                        (box[2], box[3]),
                        BBOX_GREEN,
                        1
                    )

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_window.image(frame)
