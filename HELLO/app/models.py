# app/models.py

from pydantic import BaseModel
from typing import List, Optional

# Existing Models
class SearchRequest(BaseModel):
    input_abstract: str
    filter_status: Optional[str] = "all"  # Options: "granted", "submitted", "all"
    top_k: Optional[int] = 5

class SearchResponseItem(BaseModel):
    id: int
    title: str
    abstract: str
    status: str
    similarity: float

class SearchResponse(BaseModel):
    results: List[SearchResponseItem]

# New Models
class User(BaseModel):
    id: int
    username: str
    role: str  # e.g., "user", "admin"
    tech_stack: List[str]

class Reviewer(BaseModel):
    id: int
    name: str
    expertise: List[str]
    reviewer_type: str  # "MTC" or "Patent"
    patent_review: bool
    technological_review: bool
    status: str  # "available", "busy"

class Idea(BaseModel):
    id: int
    title: str
    problem: str
    solution: str
    team_suggesting: str
    file_type: str
    sid: str
    technology: List[str]
    status: str  # e.g., "granted", "submitted"

class Assignment(BaseModel):
    assignment_id: int
    idea_id: int
    reviewer_ids_mtc: List[int]
    reviewer_ids_patent: List[int]
    status: str  # "assigned", "completed"
    comments: Optional[str] = ""
