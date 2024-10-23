# app/assignment.py

import json
import os
from typing import List, Tuple
from .models import Assignment, Idea, Reviewer

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def load_json(file_name: str):
    path = os.path.join(DATA_DIR, file_name)
    with open(path, 'r') as f:
        return json.load(f)

def save_json(file_name: str, data):
    path = os.path.join(DATA_DIR, file_name)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def auto_assign_reviewers(idea: Idea, reviewers: List[Reviewer], current_assignments: List[Assignment]) -> Tuple[List[int], List[int]]:
    """
    Assign MTC and Patent reviewers based on matching expertise with the idea's technology.
    Returns two lists: MTC reviewer IDs and Patent reviewer IDs.
    """
    matching_mtc_reviewers = []
    matching_patent_reviewers = []
    
    for reviewer in reviewers:
        if reviewer.status != "available":
            continue
        # Check if reviewer's expertise intersects with idea's technology
        if set(reviewer.expertise).intersection(set(idea.technology)):
            if reviewer.reviewer_type == "MTC":
                matching_mtc_reviewers.append(reviewer.id)
            elif reviewer.reviewer_type == "Patent":
                matching_patent_reviewers.append(reviewer.id)
    
    # Decide number of reviewers per type (e.g., 2 for MTC, 2 for Patent)
    num_mtc_reviewers = 2
    num_patent_reviewers = 2
    
    assigned_mtc_reviewers = matching_mtc_reviewers[:num_mtc_reviewers] if len(matching_mtc_reviewers) >= num_mtc_reviewers else matching_mtc_reviewers
    assigned_patent_reviewers = matching_patent_reviewers[:num_patent_reviewers] if len(matching_patent_reviewers) >= num_patent_reviewers else matching_patent_reviewers
    
    return assigned_mtc_reviewers, assigned_patent_reviewers

def create_assignment(idea_id: int, reviewer_ids_mtc: List[int], reviewer_ids_patent: List[int], assignments: List[Assignment]) -> Assignment:
    """
    Create a new assignment entry.
    """
    new_id = max([a.assignment_id for a in assignments], default=1000) + 1
    assignment = Assignment(
        assignment_id=new_id,
        idea_id=idea_id,
        reviewer_ids_mtc=reviewer_ids_mtc,
        reviewer_ids_patent=reviewer_ids_patent,
        status="assigned",
        comments=""
    )
    return assignment
