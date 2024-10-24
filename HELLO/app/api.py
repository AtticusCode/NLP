from fastapi import APIRouter, HTTPException
from typing import List
from .models import (
    SearchRequest, SearchResponse, Idea, 
    User, Reviewer, Assignment, SearchResponseItem
)
from .embedding import EmbeddingModel
from .search import AbstractMatcher
from .assignment import (
    load_json, save_json, create_assignment
)
import json
import os
import numpy as np
import logging
import yake

router = APIRouter()

# Configure logging
logger = logging.getLogger("app.api")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Initialize EmbeddingModel for topic extraction
embedding_model = EmbeddingModel()

# Paths to JSON files
DATA_PATH_ABSTRACTS = os.path.join(os.path.dirname(__file__), '..', 'data', 'ideas.json')
DATA_PATH_USERS = os.path.join(os.path.dirname(__file__), '..', 'data', 'users.json')
DATA_PATH_REVIEWERS = os.path.join(os.path.dirname(__file__), '..', 'data', 'reviewers.json')
DATA_PATH_ASSIGNMENTS = os.path.join(os.path.dirname(__file__), '..', 'data', 'assignments.json')

# Load data
def load_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_data(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

# Initialize Ideas
ideas_data = load_data(DATA_PATH_ABSTRACTS)
ideas = [Idea(**item) for item in ideas_data]

# Extract topics from all abstracts
abstract_texts = [f"{idea.problem} {idea.solution}" for idea in ideas]
topics = embedding_model.extract_topics(abstract_texts)
topic_embeddings = embedding_model.get_topic_embeddings(topics)

# Initialize matcher with topic embeddings
matcher = AbstractMatcher(ideas, topic_embeddings, embedding_model)

# Function to extract technologies using YAKE
def extract_technologies_semantic(text: str) -> List[str]:
    language = "en"
    max_ngram_size = 3
    deduplication_threshold = 0.9
    num_of_keywords = 10

    custom_kw_extractor = yake.KeywordExtractor(
        lan=language, 
        n=max_ngram_size, 
        dedupLim=deduplication_threshold, 
        top=num_of_keywords
    )
    keywords = custom_kw_extractor.extract_keywords(text)
    extracted_keywords = [kw[0] for kw in keywords]
    return extracted_keywords

# Function to compute similarity between idea topics and reviewer expertise
def compute_similarity_with_topics(idea_topics: List[str], reviewers: List[Reviewer]) -> List[dict]:
    # Initialize TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Fit and transform idea topics and reviewer expertise
    all_texts = idea_topics + [' '.join(reviewer.expertise) for reviewer in reviewers]
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Compute cosine similarity between idea topics and reviewers
    idea_tfidf = tfidf_matrix[:len(idea_topics)]
    reviewer_tfidf = tfidf_matrix[len(idea_topics):]
    
    similarity_matrix = cosine_similarity(idea_tfidf, reviewer_tfidf)
    similarities = similarity_matrix.mean(axis=0)
    
    similarity_list = [
        {"reviewer_id": reviewer.id, "similarity": similarity}
        for reviewer, similarity in zip(reviewers, similarities)
    ]
    
    return similarity_list

# Function to assign MTC reviewers
def assign_mtc_reviewers(idea_topics: List[str], reviewers: List[Reviewer]) -> List[int]:
    similarities = compute_similarity_with_topics(idea_topics, reviewers)
    similarities_sorted = sorted(similarities, key=lambda x: x["similarity"], reverse=True)
    mtc_reviewers = [sim["reviewer_id"] for sim in similarities_sorted if reviewers[sim["reviewer_id"]].reviewer_type == "MTC"]
    return mtc_reviewers[:2]  # Select top 2 MTC reviewers

# Function to assign E2 reviewers
def assign_e2_reviewers(idea_topics: List[str], reviewers: List[Reviewer]) -> List[int]:
    similarities = compute_similarity_with_topics(idea_topics, reviewers)
    similarities_sorted = sorted(similarities, key=lambda x: x["similarity"], reverse=True)
    e2_reviewers = [sim["reviewer_id"] for sim in similarities_sorted if reviewers[sim["reviewer_id"]].reviewer_type == "E2"]
    return e2_reviewers[:2]  # Select top 2 E2 reviewers

# Endpoint to get similar abstracts
@router.post("/search", response_model=SearchResponse)
def search_abstract(request: SearchRequest):
    if not request.input_abstract:
        raise HTTPException(status_code=400, detail="Input abstract is required.")
    
    input_embedding = embedding_model.encode([request.input_abstract])[0]
    results = matcher.search(request, input_embedding)
    
    return SearchResponse(results=results)

# Endpoint to assign MTC reviewers
@router.post("/assign/mtc", response_model=Assignment)
def assign_mtc(request: SearchRequest):
    idea_topics = embedding_model.extract_topics([request.input_abstract])
    reviewers_data = load_data(DATA_PATH_REVIEWERS)
    reviewers = [Reviewer(**rev) for rev in reviewers_data]
    assigned_mtc_reviewers = assign_mtc_reviewers(idea_topics, reviewers)
    return {"reviewer_ids": assigned_mtc_reviewers}

# Endpoint to assign E2 reviewers
@router.post("/assign/e2", response_model=Assignment)
def assign_e2(request: SearchRequest):
    idea_topics = embedding_model.extract_topics([request.input_abstract])
    reviewers_data = load_data(DATA_PATH_REVIEWERS)
    reviewers = [Reviewer(**rev) for rev in reviewers_data]
    assigned_e2_reviewers = assign_e2_reviewers(idea_topics, reviewers)
    return {"reviewer_ids": assigned_e2_reviewers}
