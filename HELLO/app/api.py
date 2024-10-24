# app/api.py

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
import spacy
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

router = APIRouter()

# Configure logging
logger = logging.getLogger("app.api")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Load spaCy English model with word vectors
nlp = spacy.load("en_core_web_md")

# Initialize SentenceTransformer model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

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

# Function to extract technologies semantically from text
def extract_technologies_semantic(text: str) -> List[str]:
    doc = nlp(text)
    technologies = set()
    for chunk in doc.noun_chunks:
        tech_candidate = chunk.text.lower().strip()
        technologies.add(tech_candidate)
    return list(technologies)

# Function to compute similarity between idea topics and reviewer expertise
def compute_similarity_with_topics(idea_topics: List[str], reviewers: List[Reviewer]) -> List[dict]:
    # Encode idea topics
    idea_embeddings = sentence_model.encode(idea_topics)
    
    # Concatenate all idea topic embeddings
    idea_embedding = np.mean(idea_embeddings, axis=0).reshape(1, -1)
    
    similarities = []
    for reviewer in reviewers:
        # Concatenate reviewer's expertise into a single string
        reviewer_expertise_str = ' '.join(reviewer.expertise)
        reviewer_embedding = sentence_model.encode([reviewer_expertise_str])[0].reshape(1, -1)
        
        # Compute cosine similarity
        similarity = cosine_similarity(idea_embedding, reviewer_embedding)[0][0]
        similarities.append({
            "reviewer_id": reviewer.id,
            "similarity": similarity
        })
    return similarities

# Search Endpoint
@router.post("/search", response_model=SearchResponse)
def search_abstract(request: SearchRequest):
    if not request.input_abstract:
        raise HTTPException(status_code=400, detail="Input abstract is required.")
    
    # Generate embedding for input abstract
    input_embedding = embedding_model.encode([request.input_abstract])[0]
    
    # Perform search
    results = matcher.search(request, input_embedding)
    
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
    
    # Extract technologies semantically
    extracted_technologies = extract_technologies_semantic(combined_text)
    idea.technology = extracted_technologies
    logger.info(f"Extracted technologies: {idea.technology}")
    
    # Extract topics
    extracted_topics = embedding_model.extract_topics([combined_text])
    idea_topics = extracted_topics  # List of topics for the new idea
    logger.info(f"Extracted topics: {idea_topics}")
    
    # Assign topics to the idea (you may want to store them if necessary)
    
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

# Assignment Endpoints
@router.get("/assignments", response_model=List[Assignment])
def get_assignments():
    assignments_data = load_data(DATA_PATH_ASSIGNMENTS)
    return [Assignment(**assign) for assign in assignments_data]

# Endpoint to Assign MTC Reviewers Separately (Mandatory)
@router.post("/assignments/mtc", response_model=Assignment)
def assign_mtc_reviewers(idea_id: int):
    logger.info(f"Assigning MTC reviewers for Idea ID {idea_id}")
    # Load necessary data
    ideas_data = load_data(DATA_PATH_ABSTRACTS)
    idea = next((item for item in ideas_data if item["id"] == idea_id), None)
    if not idea:
        logger.error(f"Idea with ID {idea_id} not found.")
        raise HTTPException(status_code=404, detail="Idea not found.")
    
    if idea["status"].lower() != "submitted":
        logger.error(f"Idea ID {idea_id} is not in 'submitted' status.")
        raise HTTPException(status_code=400, detail="Only submitted ideas can be assigned MTC reviewers.")
    
    reviewers_data = load_data(DATA_PATH_REVIEWERS)
    reviewers = [Reviewer(**rev) for rev in reviewers_data]
    
    assignments_data = load_data(DATA_PATH_ASSIGNMENTS)
    assignments = [Assignment(**assign) for assign in assignments_data]
    
    # Create Idea model instance
    idea_model = Idea(**idea)
    
    # Compute similarity scores based on topics
    similarities = compute_similarity_with_topics([f"{idea_model.problem} {idea_model.solution}"], reviewers)
    
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
    
    if not assigned_mtc_reviewers:
        logger.error("No available MTC reviewers found.")
        raise HTTPException(status_code=400, detail="No available MTC reviewers found.")
    
    # Create and save assignment
    new_assignment = create_assignment(idea_model.id, assigned_mtc_reviewers, [], assignments)
    assignments_data.append(new_assignment.dict())
    save_data(DATA_PATH_ASSIGNMENTS, assignments_data)
    logger.info(f"Created new assignment ID {new_assignment.assignment_id} for Idea ID {idea_model.id} with MTC reviewers {assigned_mtc_reviewers}")
    
    # Update reviewers' status to 'busy'
    for reviewer in reviewers:
        if reviewer.id in assigned_mtc_reviewers:
            for rev in reviewers_data:
                if rev["id"] == reviewer.id:
                    rev["status"] = "busy"
    save_data(DATA_PATH_REVIEWERS, reviewers_data)
    logger.info(f"Updated status of assigned MTC reviewers to 'busy'")
    
    return new_assignment

# Endpoint to Assign Patent Reviewers Separately (Mandatory)
@router.post("/assignments/patent", response_model=Assignment)
def assign_patent_reviewers(idea_id: int):
    logger.info(f"Assigning Patent reviewers for Idea ID {idea_id}")
    # Load necessary data
    ideas_data = load_data(DATA_PATH_ABSTRACTS)
    idea = next((item for item in ideas_data if item["id"] == idea_id), None)
    if not idea:
        logger.error(f"Idea with ID {idea_id} not found.")
        raise HTTPException(status_code=404, detail="Idea not found.")
    
    if idea["status"].lower() != "submitted":
        logger.error(f"Idea ID {idea_id} is not in 'submitted' status.")
        raise HTTPException(status_code=400, detail="Only submitted ideas can be assigned Patent reviewers.")
    
    reviewers_data = load_data(DATA_PATH_REVIEWERS)
    reviewers = [Reviewer(**rev) for rev in reviewers_data]
    
    assignments_data = load_data(DATA_PATH_ASSIGNMENTS)
    assignments = [Assignment(**assign) for assign in assignments_data]
    
    # Create Idea model instance
    idea_model = Idea(**idea)
    
    # Compute similarity scores based on topics
    similarities = compute_similarity_with_topics([f"{idea_model.problem} {idea_model.solution}"], reviewers)
    
    # Sort reviewers by similarity in descending order
    similarities_sorted = sorted(similarities, key=lambda x: x["similarity"], reverse=True)
    
    # Select top K Patent reviewers
    top_k = 2
    assigned_patent_reviewers = []
    for sim in similarities_sorted:
        reviewer = next((r for r in reviewers if r.id == sim["reviewer_id"]), None)
        if reviewer and reviewer.reviewer_type.lower() == "patent" and reviewer.status.lower() == "available":
            assigned_patent_reviewers.append(reviewer.id)
            if len(assigned_patent_reviewers) >= top_k:
                break
    
    logger.info(f"Assigned Patent Reviewer IDs: {assigned_patent_reviewers}")
    
    if not assigned_patent_reviewers:
        logger.error("No available Patent reviewers found.")
        raise HTTPException(status_code=400, detail="No available Patent reviewers found.")
    
    # Create and save assignment
    new_assignment = create_assignment(idea_model.id, [], assigned_patent_reviewers, assignments)
    assignments_data.append(new_assignment.dict())
    save_data(DATA_PATH_ASSIGNMENTS, assignments_data)
    logger.info(f"Created new assignment ID {new_assignment.assignment_id} for Idea ID {idea_model.id} with Patent reviewers {assigned_patent_reviewers}")
    
    # Update reviewers' status to 'busy'
    for reviewer in reviewers:
        if reviewer.id in assigned_patent_reviewers:
            for rev in reviewers_data:
                if rev["id"] == reviewer.id:
                    rev["status"] = "busy"
    save_data(DATA_PATH_REVIEWERS, reviewers_data)
    logger.info(f"Updated status of assigned Patent reviewers to 'busy'")
    
    return new_assignment

# Additional Endpoint: Assign E2 Reviewers (Assuming E2 is another reviewer type)
@router.post("/assignments/e2", response_model=Assignment)
def assign_e2_reviewers(idea_id: int):
    logger.info(f"Assigning E2 reviewers for Idea ID {idea_id}")
    # Load necessary data
    ideas_data = load_data(DATA_PATH_ABSTRACTS)
    idea = next((item for item in ideas_data if item["id"] == idea_id), None)
    if not idea:
        logger.error(f"Idea with ID {idea_id} not found.")
        raise HTTPException(status_code=404, detail="Idea not found.")
    
    if idea["status"].lower() != "submitted":
        logger.error(f"Idea ID {idea_id} is not in 'submitted' status.")
        raise HTTPException(status_code=400, detail="Only submitted ideas can be assigned E2 reviewers.")
    
    reviewers_data = load_data(DATA_PATH_REVIEWERS)
    reviewers = [Reviewer(**rev) for rev in reviewers_data]
    
    assignments_data = load_data(DATA_PATH_ASSIGNMENTS)
    assignments = [Assignment(**assign) for assign in assignments_data]
    
    # Create Idea model instance
    idea_model = Idea(**idea)
    
    # Compute similarity scores based on topics
    similarities = compute_similarity_with_topics([f"{idea_model.problem} {idea_model.solution}"], reviewers)
    
    # Sort reviewers by similarity in descending order
    similarities_sorted = sorted(similarities, key=lambda x: x["similarity"], reverse=True)
    
    # Select top K E2 reviewers (assuming E2 is another type)
    top_k = 2
    assigned_e2_reviewers = []
    for sim in similarities_sorted:
        reviewer = next((r for r in reviewers if r.id == sim["reviewer_id"]), None)
        if reviewer and reviewer.reviewer_type.lower() == "e2" and reviewer.status.lower() == "available":
            assigned_e2_reviewers.append(reviewer.id)
            if len(assigned_e2_reviewers) >= top_k:
                break
    
    logger.info(f"Assigned E2 Reviewer IDs: {assigned_e2_reviewers}")
    
    if not assigned_e2_reviewers:
        logger.error("No available E2 reviewers found.")
        raise HTTPException(status_code=400, detail="No available E2 reviewers found.")
    
    # Create and save assignment
    new_assignment = create_assignment(idea_model.id, [], assigned_e2_reviewers, assignments)
    assignments_data.append(new_assignment.dict())
    save_data(DATA_PATH_ASSIGNMENTS, assignments_data)
    logger.info(f"Created new assignment ID {new_assignment.assignment_id} for Idea ID {idea_model.id} with E2 reviewers {assigned_e2_reviewers}")
    
    # Update reviewers' status to 'busy'
    for reviewer in reviewers:
        if reviewer.id in assigned_e2_reviewers:
            for rev in reviewers_data:
                if rev["id"] == reviewer.id:
                    rev["status"] = "busy"
    save_data(DATA_PATH_REVIEWERS, reviewers_data)
    logger.info(f"Updated status of assigned E2 reviewers to 'busy'")
    
    return new_assignment
