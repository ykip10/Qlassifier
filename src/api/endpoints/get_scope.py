from fastapi import APIRouter
from scope import SUBJECTS

router = APIRouter()


@router.get("/subjects")
def get_subjects():
    return SUBJECTS
