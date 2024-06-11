import faiss
import numpy as np


def create_faiss_index(embeddings, metadata):
    d = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(d)
    faiss_index.add(np.array(embeddings).astype('float32'))
    faiss_index.metadata = metadata
    return faiss_index

def search_in_faiss(faiss_index, embedding, k=5):
    embedding = np.array(embedding).astype('float32')
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)
    distances, indices = faiss_index.search(embedding, k)
    return distances, indices