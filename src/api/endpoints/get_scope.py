# main.py or endpoints file
from fastapi import APIRouter
from scope import SUBJECTS  # make sure scope.py is at root and has SUBJECTS list

router = APIRouter()


@router.get("/subjects")
def get_subjects():
    return SUBJECTS
