""" Runs partial or full pipeline for examination/study-designs 
downloading -> parsing -> preprocessing -> model output.

Usage: 
    python3 -m src.runPipeline subject_name year [tf-idf|instructor]         | If you want to download a VCAA exam first
    python3 -m src.runPipeline subject_name exam_path [tf-idf|instructor]    | If you want to parse a custom exam 

Last argument controls prediction methodology. 
"""
import sys 
from pathlib import Path

from src.Qlassifier.prediction import InstructPredictor, TfIdfPredictor
from src.extractor import material_collector as mc
from src.paths import DATA_DIR


def run_pipeline(
    argv: list[str] | None = None
) -> list[list[tuple[int, float]]] :
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 3:
        print(__doc__)
        return 2
    
    subject = argv[0].lower()
    if ".pdf" in argv[1]:
        # Path supplied, can just run pipeline on this exam
        path = Path(argv[1])
    else:
        # Need to download data
        try: 
            year = argv[1]
        except IndexError:
            print("If you supply a subject name, you must also supply a year. ")
            return 1 
        if "math" in subject: 
            exam_num = input("Which examination number? ")
            path = DATA_DIR / subject / "past_exams" / f"{year}_{exam_num}.pdf"
        else: 
            path = DATA_DIR / subject / "past_exams" / f"{year}.pdf"
        # extract all required data
        mc.main(argv = [subject, year])
    
    sd_path = path.parents[1] / "study_design" / f"{subject}_sd.docx"
    # Running either a tf-idf or instructor exclusive model
    if argv[2] == "tf-idf":
        pred = TfIdfPredictor(sd_path, subject=subject)
    elif argv[2] == "instructor":
        pred = InstructPredictor(sd_path, subject=subject)
    else: 
        print("Third argument must be either 'tf-idf' or 'instructor.'")
        return 1
    
    results = pred.run(path)
    results.print_top3s()
    return 0


if __name__ == "__main__":
	raise SystemExit(run_pipeline())