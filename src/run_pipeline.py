""" Runs partial or full pipeline for examination/study-designs 
downloading -> parsing -> preprocessing -> model output.

Usage: 
    python3 -m src.run_pipeline subject_name year [tf-idf|instructor]         | If you want to download a VCAA exam first
    python3 -m src.run_pipeline subject_name exam_path [tf-idf|instructor]    | If you want to parse a custom exam 

The tf-idf/instructor argument can be appended to get solely tf-idf/instructor predictions 
instead of using a combined model. 
"""
import sys 
from pathlib import Path

from src.Qlassifier.prediction import run_instructor, run_combined, run_tf_idf
from src.loader import material_collector as mc
from src.paths import DATA_DIR


def run_pipeline(
    argv: list[str] | None = None
) -> list[list[tuple[int, float]]] :
    argv = argv if argv is not None else sys.argv[1:]
    if not argv or len(argv) > 3:
        print(__doc__)
        return 2
    
    subject = argv[0]
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

    if len(argv) == 2: 
        # Run combined model
        results = run_combined(path, subject, tf_idf_weight=0.3)
        results.print_top3s()
        return 0
    
    # Running either a tf-idf or instructor exclusive model
    if argv[2] == "tf-idf":
        results = run_tf_idf(path, subject, include_report=False)
    elif argv[2] == "instructor":
        results = run_instructor(path, subject)
    else: 
        print("Third argument must be either empty or equal to 'tf-idf.'")
        return 1
    
    results.print_top3s()
    return 0


if __name__ == "__main__":
	raise SystemExit(run_pipeline())