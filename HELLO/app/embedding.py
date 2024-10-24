# app/embedding.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import numpy as np
from typing import List

class EmbeddingModel:
    def __init__(self, corpus_texts: List[str]):
        # Initialize TF-IDF Vectorizer and NMF for topic modeling
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus_texts)
        
        # Fit NMF model
        self.nmf_model = NMF(n_components=10, random_state=42)
        self.nmf_features = self.nmf_model.fit_transform(self.tfidf_matrix)
        
        # Store feature names
        self.feature_names = self.vectorizer.get_feature_names_out()
    
    def encode(self, texts: List[str]) -> np.ndarray:
        # Transform texts using the fitted TF-IDF vectorizer
        return self.vectorizer.transform(texts).toarray()
    
    def extract_topics(self, texts: List[str]) -> List[str]:
        # Transform texts using the fitted TF-IDF vectorizer
        tfidf = self.vectorizer.transform(texts)
        # Transform TF-IDF vectors using the fitted NMF model
        nmf_features = self.nmf_model.transform(tfidf)
        topics = []
        for topic_idx, topic in enumerate(self.nmf_model.components_):
            top_features_ind = topic.argsort()[:-11:-1]
            top_features = [self.feature_names[i] for i in top_features_ind]
            topics.append(" ".join(top_features))
        return topics
    
    def get_topic_embeddings(self, topics: List[str]) -> np.ndarray:
        # Transform topics using the fitted TF-IDF vectorizer
        return self.encode(topics)
