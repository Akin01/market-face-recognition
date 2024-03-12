from typing import List

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from config.database import QDRANT_HOST, QDRANT_PORT, VECTOR_SIZE, QDRANT_COLLECTION_NAME


def get_qdrant_client():
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def initiate_qdrant_collection(*, distance_option: Distance = Distance.COSINE):
    client = get_qdrant_client()
    client.recreate_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=distance_option),
    )


def upload_embedding(*, vector_data: List[PointStruct]):
    client = get_qdrant_client()
    return client.upsert(
        collection_name=QDRANT_COLLECTION_NAME,
        wait=True,
        points=vector_data,
    )
