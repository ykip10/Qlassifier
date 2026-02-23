from pathlib import Path

from fastapi import File, UploadFile, APIRouter, Form

from src.runPipeline import run_model_pipeline

router = APIRouter()


@router.post("/run_instructor")
async def run_instructor_endpoint(
    subject: str = Form(...), file: UploadFile = File(...)
):
    contents = await file.read()

    tmp_path = f"/tmp/{file.filename}"
    with open(tmp_path, "wb") as f:
        f.write(contents)
    subject = "_".join([word.lower() for word in subject.split()])
    sd_path = Path("data") / subject / "study_design" / f"{subject}_sd.docx"
    res = run_model_pipeline(tmp_path, subject, model="instructor", sd_path=sd_path)
    out = res.build_top3s(ndigits=3)
    return out
