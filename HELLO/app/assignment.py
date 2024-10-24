# app/assignment.py

import json
import os
from typing import List
from .models import Assignment

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def load_json(file_name: str):
    """
    Loads JSON data from the specified file.
    """
    path = os.path.join(DATA_DIR, file_name)
    with open(path, 'r') as f:
        return json.load(f)

def save_json(file_name: str, data):
    """
    Saves JSON data to the specified file.
    """
    path = os.path.join(DATA_DIR, file_name)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def create_assignment(idea_id: int, reviewer_ids_mtc: List[int], reviewer_ids_e2: List[int], assignments: List[Assignment]) -> Assignment:
    """
    Creates a new assignment entry with a unique assignment_id.
    """
    new_id = max([a.assignment_id for a in assignments], default=1000) + 1
    assignment = Assignment(
        assignment_id=new_id,
        idea_id=idea_id,
        reviewer_ids_mtc=reviewer_ids_mtc,
        reviewer_ids_e2=reviewer_ids_e2,
        reviewer_ids_patent=[],  # Assuming Patent reviewers are handled separately
        status="assigned",
        comments=""
    )
    return assignment
