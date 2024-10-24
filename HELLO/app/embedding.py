# app/embedding.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import numpy as np
from typing import List

class EmbeddingModel:
    def __init__(self, corpus_texts: List[str]):
        """
        Initializes the TF-IDF Vectorizer and NMF model, fitting them on the corpus.
        """
        # Initialize TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus_texts)
        
        # Initialize and fit NMF model
        self.nmf_model = NMF(n_components=10, random_state=42)
        self.nmf_features = self.nmf_model.fit_transform(self.tfidf_matrix)
        
        # Store feature names for topic interpretation
        self.feature_names = self.vectorizer.get_feature_names_out()
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Transforms new texts into TF-IDF vectors using the fitted vectorizer.
        """
        return self.vectorizer.transform(texts).toarray()
    
    def extract_topics(self, texts: List[str]) -> List[str]:
        """
        Extracts dominant topics from the given texts using the fitted NMF model.
        Each text is associated with its dominant topic.
        """
        tfidf = self.vectorizer.transform(texts)
        nmf_features = self.nmf_model.transform(tfidf)
        topics = []
        for feature in nmf_features:
            # Identify the dominant topic
            topic_idx = feature.argmax()
            topic = self.nmf_model.components_[topic_idx]
            # Get top 10 words for the topic
            top_features_ind = topic.argsort()[:-11:-1]
            top_features = [self.feature_names[i] for i in top_features_ind]
            topics.append(" ".join(top_features))
        return topics
    
    def get_topic_embeddings(self, topics: List[str]) -> np.ndarray:
        """
        Encodes topic strings into TF-IDF vectors.
        """
        return self.encode(topics)
