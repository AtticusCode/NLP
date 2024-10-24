# app/embedding.py

from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingModel:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
