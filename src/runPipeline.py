""" Runs partial or full pipeline for examination/study-designs 
downloading -> parsing -> preprocessing -> model output.

Usage: 
    python3 -m src.runPipeline subject_name year [tf-idf|instructor]         | If you want to download a VCAA exam first
    python3 -m src.runPipeline subject_name exam_path [tf-idf|instructor]    | If you want to parse a custom exam 

Last argument controls prediction methodology. 
"""
from __future__ import annotations
import sys 
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from src.Qlassifier.prediction import InstructPredictor, TfIdfPredictor
from src.extractor import materialCollector as mc
from src.paths import DATA_DIR

if TYPE_CHECKING:
    from os import PathLike
    from src.Qlassifier.results import Results


def run_model_pipeline(
    path: str | PathLike[str],
    subject: str,
    model: Literal["tf-idf", "instructor"] | InstructPredictor | TfIdfPredictor,
    sd_path: str | PathLike[str] | None = None,
) -> Results:
    """ Runs modelling pipeline. Either loads the model or uses a pre-loaded one to
    generate question labels on an exam at `path`.

    Parameters
    ----------
    path   : Path to examination we want to generate labels for.
    subject: Relevant subject name.
    model  : Either a string literal (if the model is to be loaded) or a pre-loaded model
             instance. 
    sd_path: Relevant subject study designs' path. If `None` provided, attempts to find it.

    Returns
    -------
    `Results` object. 
    """
    path = Path(path)
    # Try to load a path if None given
    sd_path = Path(sd_path) if sd_path is not None else path.parents[1] / "study_design" / f"{subject}_sd.docx"
    print(sd_path)
    if not sd_path.exists():
        raise ValueError("Unable to load a study design. Please provide a path.")
    
    pred = model
    if isinstance(model, str):
        if model == "tf-idf":
            pred = TfIdfPredictor(sd_path, subject=subject)
        elif model == "instructor":
            pred = InstructPredictor(sd_path, subject)
        else: 
            raise ValueError("If `model` is given as a string, it must be one of `tf-idf` or `instructor`.")

    res = pred.run(path)
    return res


def main(
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
    
    res = run_model_pipeline(path, subject, argv[2])
    res.print_top3s()
    return 0


if __name__ == "__main__":
	raise SystemExit(main())