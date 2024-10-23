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
    load_json, save_json, auto_assign_reviewers, create_assignment
)
import json
import os
import numpy as np

router = APIRouter()

# Paths to JSON files
DATA_PATH_ABSTRACTS = os.path.join(os.path.dirname(__file__), '..', 'data', 'abstracts.json')
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

# Create abstract texts by combining problem and solution
abstract_texts = [f"{idea.problem} {idea.solution}" for idea in ideas]

# Initialize embedding model and generate embeddings
embedding_model = EmbeddingModel()
embeddings = embedding_model.encode(abstract_texts)

# Initialize matcher
matcher = AbstractMatcher(ideas, embeddings)

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
    ideas_data = load_data(DATA_PATH_ABSTRACTS)
    if any(i["id"] == idea.id for i in ideas_data):
        raise HTTPException(status_code=400, detail="Idea with this ID already exists.")
    ideas_data.append(idea.dict())
    save_data(DATA_PATH_ABSTRACTS, ideas_data)
    
    # Auto-assign reviewers
    reviewers_data = load_data(DATA_PATH_REVIEWERS)
    reviewers = [Reviewer(**rev) for rev in reviewers_data]
    assignments_data = load_data(DATA_PATH_ASSIGNMENTS)
    assignments = [Assignment(**assign) for assign in assignments_data]
    
    assigned_mtc_ids, assigned_patent_ids = auto_assign_reviewers(idea, reviewers, assignments)
    if assigned_mtc_ids or assigned_patent_ids:
        new_assignment = create_assignment(idea.id, assigned_mtc_ids, assigned_patent_ids, assignments)
        assignments_data.append(new_assignment.dict())
        save_data(DATA_PATH_ASSIGNMENTS, assignments_data)
        
        # Update reviewers' status to 'busy'
        for reviewer in reviewers:
            if reviewer.id in assigned_mtc_ids or reviewer.id in assigned_patent_ids:
                for rev in reviewers_data:
                    if rev["id"] == reviewer.id:
                        rev["status"] = "busy"
        save_data(DATA_PATH_REVIEWERS, reviewers_data)
    
    return idea

# Assignment Endpoints
@router.get("/assignments", response_model=List[Assignment])
def get_assignments():
    assignments_data = load_data(DATA_PATH_ASSIGNMENTS)
    return [Assignment(**assign) for assign in assignments_data]

# Separate Endpoint for MTC Reviewer Assignments
@router.post("/assignments/mtc", response_model=Assignment)
def assign_mtc_reviewers(idea_id: int):
    # Load necessary data
    ideas_data = load_data(DATA_PATH_ABSTRACTS)
    idea = next((item for item in ideas_data if item["id"] == idea_id), None)
    if not idea:
        raise HTTPException(status_code=404, detail="Idea not found.")
    
    reviewers_data = load_data(DATA_PATH_REVIEWERS)
    reviewers = [Reviewer(**rev) for rev in reviewers_data]
    
    assignments_data = load_data(DATA_PATH_ASSIGNMENTS)
    assignments = [Assignment(**assign) for assign in assignments_data]
    
    # Create Idea model instance
    idea_model = Idea(**idea)
    
    # Assign MTC reviewers
    assigned_mtc_ids, _ = auto_assign_reviewers(idea_model, reviewers, assignments)
    
    if not assigned_mtc_ids:
        raise HTTPException(status_code=400, detail="No available MTC reviewers found.")
    
    # Create and save assignment
    new_assignment = create_assignment(idea_model.id, assigned_mtc_ids, [], assignments)
    assignments_data.append(new_assignment.dict())
    save_data(DATA_PATH_ASSIGNMENTS, assignments_data)
    
    # Update reviewers' status to 'busy'
    for reviewer in reviewers:
        if reviewer.id in assigned_mtc_ids:
            for rev in reviewers_data:
                if rev["id"] == reviewer.id:
                    rev["status"] = "busy"
    save_data(DATA_PATH_REVIEWERS, reviewers_data)
    
    return new_assignment

# Separate Endpoint for Patent Reviewer Assignments
@router.post("/assignments/patent", response_model=Assignment)
def assign_patent_reviewers(idea_id: int):
    # Load necessary data
    ideas_data = load_data(DATA_PATH_ABSTRACTS)
    idea = next((item for item in ideas_data if item["id"] == idea_id), None)
    if not idea:
        raise HTTPException(status_code=404, detail="Idea not found.")
    
    reviewers_data = load_data(DATA_PATH_REVIEWERS)
    reviewers = [Reviewer(**rev) for rev in reviewers_data]
    
    assignments_data = load_data(DATA_PATH_ASSIGNMENTS)
    assignments = [Assignment(**assign) for assign in assignments_data]
    
    # Create Idea model instance
    idea_model = Idea(**idea)
    
    # Assign Patent reviewers
    _, assigned_patent_ids = auto_assign_reviewers(idea_model, reviewers, assignments)
    
    if not assigned_patent_ids:
        raise HTTPException(status_code=400, detail="No available Patent reviewers found.")
    
    # Create and save assignment
    new_assignment = create_assignment(idea_model.id, [], assigned_patent_ids, assignments)
    assignments_data.append(new_assignment.dict())
    save_data(DATA_PATH_ASSIGNMENTS, assignments_data)
    
    # Update reviewers' status to 'busy'
    for reviewer in reviewers:
        if reviewer.id in assigned_patent_ids:
            for rev in reviewers_data:
                if rev["id"] == reviewer.id:
                    rev["status"] = "busy"
    save_data(DATA_PATH_REVIEWERS, reviewers_data)
    
    return new_assignment
