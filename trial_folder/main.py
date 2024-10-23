# main.py

from fastapi import FastAPI
from app.api import router
import uvicorn

app = FastAPI(
    title="NLP-based Abstract Matching System",
    description="Finds similar abstracts based on semantic similarity with filtering and reviewer assignment options.",
    version="1.2.0"
)

app.include_router(router, prefix="/api")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
