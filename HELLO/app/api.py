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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import yake  # Import YAKE

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
    max_ngram_size = 3  # Adjust this as needed
    deduplication_threshold = 0.9
    num_of_keywords = 10  # Adjust the number of keywords

    custom_kw_extractor = yake.KeywordExtractor(
        lan=language, 
        n=max_ngram_size, 
        dedupLim=deduplication_threshold, 
        top=num_of_keywords
    )
    keywords = custom_kw_extractor.extract_keywords(text)
    extracted_keywords = [kw[0] for kw in keywords]  # Extract only the keywords
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
    
    # Aggregate similarity scores for each reviewer
    similarities = similarity_matrix.mean(axis=0)
    
    similarity_list = [
        {"reviewer_id": reviewer.id, "similarity": similarity}
        for reviewer, similarity in zip(reviewers, similarities)
    ]
    
    return similarity_list

# Search Endpoint
@router.post("/search", response_model=SearchResponse)
def search_abstract(request: SearchRequest):
    if not request.input_abstract:
        raise HTTPException(status_code=400, detail="Input abstract is required.")
    
    # Generate TF-IDF vector for input abstract
    input_vector = embedding_model.vectorizer.transform([request.input_abstract])
    
    # Perform search using NMF components
    results = matcher.search(request, input_vector)
    
    return SearchResponse(results=results)

# User Endpoints
@router.get("/users", response_model=List[User])
def get_users():
    users_data = load_data(DATA_PATH_USERS)
    return [User(**user) for user in users_data]

@router.post("/users", response_model=User)
def add_user(user: User):
    users_data = load_data(DATA_PATH_USERS)
    if any(u["id"] == user.id for u in users_data):
        raise HTTPException(status_code=400, detail="User with this ID already exists.")
    users_data.append(user.dict())
    save_data(DATA_PATH_USERS, users_data)
    return user

# Reviewer Endpoints
@router.get("/reviewers", response_model=List[Reviewer])
def get_reviewers():
    reviewers_data = load_data(DATA_PATH_REVIEWERS)
    return [Reviewer(**rev) for rev in reviewers_data]

@router.post("/reviewers", response_model=Reviewer)
def add_reviewer(reviewer: Reviewer):
    reviewers_data = load_data(DATA_PATH_REVIEWERS)
    if any(r["id"] == reviewer.id for r in reviewers_data):
        raise HTTPException(status_code=400, detail="Reviewer with this ID already exists.")
    reviewers_data.append(reviewer.dict())
    save_data(DATA_PATH_REVIEWERS, reviewers_data)
    return reviewer

# Idea Endpoints
@router.get("/ideas", response_model=List[Idea])
def get_ideas():
    ideas_data = load_data(DATA_PATH_ABSTRACTS)
    return [Idea(**idea) for idea in ideas_data]

@router.post("/ideas", response_model=Idea)
def add_idea(idea: Idea):
    logger.info(f"Received new idea submission: {idea.title} (ID: {idea.id})")
    ideas_data = load_data(DATA_PATH_ABSTRACTS)
    if any(i["id"] == idea.id for i in ideas_data):
        logger.error(f"Idea with ID {idea.id} already exists.")
        raise HTTPException(status_code=400, detail="Idea with this ID already exists.")
    
    # Combine problem and solution to form the abstract
    combined_text = f"{idea.problem} {idea.solution}"
    
    # Extract technologies using YAKE
    extracted_technologies = extract_technologies_semantic(combined_text)
    idea.technology = extracted_technologies
    logger.info(f"Extracted technologies: {idea.technology}")
    
    # Extract topics using NMF
    extracted_topics = embedding_model.extract_topics([combined_text])
    idea_topics = extracted_topics  # List of topics for the new idea
    logger.info(f"Extracted topics: {idea_topics}")
    
    # Append the idea with extracted technologies
    ideas_data.append(idea.dict())
    save_data(DATA_PATH_ABSTRACTS, ideas_data)
    logger.info(f"Saved idea ID {idea.id} to ideas.json")
    
    # Auto-assign reviewers based on topics
    if idea.status.lower() == "submitted":
        logger.info(f"Auto-assigning reviewers for Idea ID {idea.id}")
        reviewers_data = load_data(DATA_PATH_REVIEWERS)
        reviewers = [Reviewer(**rev) for rev in reviewers_data]
        assignments_data = load_data(DATA_PATH_ASSIGNMENTS)
        assignments = [Assignment(**assign) for assign in assignments_data]
        
        # Compute similarity scores based on topics
        similarities = compute_similarity_with_topics(idea_topics, reviewers)
        
        # Sort reviewers by similarity in descending order
        similarities_sorted = sorted(similarities, key=lambda x: x["similarity"], reverse=True)
        
        # Select top K MTC reviewers
        top_k = 2
        assigned_mtc_reviewers = []
        for sim in similarities_sorted:
            reviewer = next((r for r in reviewers if r.id == sim["reviewer_id"]), None)
            if reviewer and reviewer.reviewer_type.lower() == "mtc" and reviewer.status.lower() == "available":
                assigned_mtc_reviewers.append(reviewer.id)
                if len(assigned_mtc_reviewers) >= top_k:
                    break
        
        logger.info(f"Assigned MTC Reviewer IDs: {assigned_mtc_reviewers}")
        
        if assigned_mtc_reviewers:
            new_assignment = create_assignment(idea.id, assigned_mtc_reviewers, [], assignments)
            assignments_data.append(new_assignment.dict())
            save_data(DATA_PATH_ASSIGNMENTS, assignments_data)
            logger.info(f"Created new assignment for Idea ID {idea.id} with MTC reviewers {assigned_mtc_reviewers}")
            
            # Update reviewers' status to 'busy'
            for reviewer in reviewers:
                if reviewer.id in assigned_mtc_reviewers:
                    for rev in reviewers_data:
                        if rev["id"] == reviewer.id:
                            rev["status"] = "busy"
            save_data(DATA_PATH_REVIEWERS, reviewers_data)
            logger.info(f"Updated status of assigned MTC reviewers to 'busy'")
    
    return idea
