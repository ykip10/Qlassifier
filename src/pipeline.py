""" Runs partial or full pipeline for examination/study-designs 
downloading -> parsing -> preprocessing -> model output.

Usage: 
    python3 -m src.pipeline subject_name year         | If you want to download a VCAA exam
    python3 -m src.pipeline subject_name exam_path    | If you want to parse a custom exam 
"""
import sys 
from pathlib import Path

from src.Qlassifier.prediction import run_instructor
from src.loader import material_collector as mc
from src.paths import DATA_DIR

def run_pipeline(
    argv: list[str] | None = None
) -> list[list[tuple[int, float]]] :
    argv = argv if argv is not None else sys.argv[1:]
    if not argv or len(argv) > 2:
        print(__doc__)
        return 2
    
    subject = argv[0]
    if ".pdf" in argv[1]:
        path = Path(argv[1])
    else:
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
    
    top_3s, qns = run_instructor(path, subject)
    for qn, top3 in zip(qns, top_3s):
        top_3_rounded = [(idx, round(sim, 3)) for idx, sim in top3]
        print(f"{qn}: {top_3_rounded}")


if __name__ == "__main__":
	raise SystemExit(run_pipeline())