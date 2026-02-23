from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.endpoints.embeddings import router as instructor_router
from src.api.endpoints.get_scope import router as subject_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(instructor_router)
app.include_router(subject_router)

# Serve static files
app.mount("/static", StaticFiles(directory="src/api/static"), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse("src/api/static/index.html")
