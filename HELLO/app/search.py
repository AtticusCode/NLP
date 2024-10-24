# app/search.py

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List
from .models import Idea, SearchRequest, SearchResponseItem
from .embedding import EmbeddingModel

class AbstractMatcher:
    def __init__(self, ideas: List[Idea], embeddings: np.ndarray, embedding_model: EmbeddingModel):
        self.ideas = ideas
        self.embeddings = embeddings
        self.embedding_model = embedding_model

    def search(self, request: SearchRequest, input_embedding: np.ndarray) -> List[SearchResponseItem]:
        # Calculate cosine similarity
        similarities = cosine_similarity([input_embedding], self.embeddings)[0]
        
        # Create list of tuples (index, similarity)
        similarity_scores = list(enumerate(similarities))
        
        # Apply filter based on status
        if request.filter_status.lower() in ["granted", "submitted"]:
            filtered = [
                (idx, score) for idx, score in similarity_scores
                if self.ideas[idx].status.lower() == request.filter_status.lower()
            ]
        else:
            filtered = similarity_scores  # "all"

        # Sort based on similarity score in descending order
        sorted_similarities = sorted(filtered, key=lambda x: x[1], reverse=True)

        # Get top K
        top_k = request.top_k
        top_results = sorted_similarities[:top_k]

        # Prepare response
        results = []
        for idx, score in top_results:
            idea = self.ideas[idx]
            result_item = SearchResponseItem(
                id=idea.id,
                title=idea.title,
                abstract=f"{idea.problem} {idea.solution}",
                status=idea.status,
                similarity=round(float(score), 4)
            )
            results.append(result_item)

        return results
