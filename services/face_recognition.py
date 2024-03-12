from typing import Union

from deepface import DeepFace
import numpy as np
from enum import Enum
from qdrant_client import QdrantClient
from services.database import QDRANT_COLLECTION_NAME, get_qdrant_client


class ModelName(Enum):
    OpenFace = "OpenFace"
    DeepFace = "DeepFace"
    DeepID = "DeepID"
    Dlib = "Dlib"
    ArcFace = "ArcFace"
    SFace = "SFace"
    VGGFace = "VGG-Face"
    Facenet = "Facenet"
    Facenet512 = "Facenet512"


class FaceRecognition:
    image: np.ndarray | None = None
    model_name: str = ModelName.Facenet.value
    qdrant_client: QdrantClient

    def __init__(self):
        self.qdrant_client = get_qdrant_client()

    def generate_embedding(self, image: np.ndarray, model: Union[ModelName, None] = None):
        if model is None:
            model = self.model_name
        else:
            model = model.value

        face_embedding = DeepFace.represent(
            image,
            model_name=model,
            enforce_detection=True,
            detector_backend="skip"
        )

        return face_embedding[0]["embedding"]

    def search_face(self, *, image_vector: np.ndarray, score_threshold: int = 0.85):
        return self.qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=image_vector,
            with_payload=True,
            score_threshold=score_threshold,
            limit=1
        )
