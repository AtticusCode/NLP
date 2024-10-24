from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import numpy as np
from typing import List

class EmbeddingModel:
    def __init__(self):
        # Initialize TF-IDF Vectorizer and NMF for topic modeling
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.nmf_model = NMF(n_components=10, random_state=42)  # Adjust n_components as needed

    def encode(self, texts: list) -> np.ndarray:
        # Vectorize text using TF-IDF
        return self.vectorizer.fit_transform(texts).toarray()

    def extract_topics(self, texts: list) -> List[str]:
        # Apply TF-IDF and NMF for topic extraction
        tfidf = self.vectorizer.fit_transform(texts)
        nmf_features = self.nmf_model.fit_transform(tfidf)
        feature_names = self.vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(self.nmf_model.components_):
            top_features_ind = topic.argsort()[:-11:-1]
            top_features = [feature_names[i] for i in top_features_ind]
            topics.append(" ".join(top_features))
        return topics

    def get_topic_embeddings(self, topics: list) -> np.ndarray:
        # Vectorize topics using TF-IDF
        return self.encode(topics)
