""" Runs partial or full pipeline for examination/study-designs 
downloading -> parsing -> preprocessing -> model output.

Usage: 
    python3 -m src.pipeline subject_name year | If you want to download a VCAA exam
    python3 -m src.pipeline exam_path         | If you want to parse a custom exam 
"""
import sys 

from src.Qlassifier.prediction import run_instructor
from src.loader import material_collector as mc

def main():
    argv = sys.argv[1:]
    if not argv or len(argv) > 2:
        print(__doc__)
        return 2
    
    if ".pdf" in argv[0]:
        path = argv[0]
    else:
        subject = argv[0]
        try: 
            year = argv[1]
        except IndexError:
            print("If you supply a subject name, you must also supply a year. ")
            return 1 

        # extract all required data
        mc.main(argv = [subject, year])
    
    run_instructor(path)


if __name__ == "__main__":
	raise SystemExit(main())