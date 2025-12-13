from fastapi import FastAPI

from src.api.endpoints.embeddings import router as instructor_router

app = FastAPI()

app.include_router(instructor_router)

