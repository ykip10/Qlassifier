from collections import defaultdict as dd 

from fastapi import File, UploadFile, APIRouter

from src.run_pipeline import run_pipeline

router = APIRouter()

@router.post("/run_instructor")
async def run_instructor_endpoint(subject: str, file: UploadFile = File(...)):
    contents = await file.read()

    tmp_path = f"/tmp/{file.filename}"
    with open(tmp_path, "wb") as f:
        f.write(contents)

    top3s, qns = run_instructor(tmp_path, subject, model = instructor)
    out = dd(list)
    for qn, top3 in zip(qns, top3s):
        top3_rounded = [(idx, round(sim, 3)) for idx, sim in top3]
        out[qn] = top3_rounded
    return out