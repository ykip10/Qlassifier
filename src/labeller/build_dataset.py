""" Builds the dataset to be labelled.  Goes through every subject and year in scope
 and compiles a semi-labelled dataset
"""
from itertools import product
from typing import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from InstructorEmbedding import INSTRUCTOR
from sentence_transformers import SentenceTransformer

from src.Qlassifier.results import Results
from src.Qlassifier.prediction import (SentenceTransPredictor,
                                       InstructPredictor,\
                                        TfIdfPredictor)
from src.extractor import material_collector as mc
import scope
from src.paths import DATA_DIR


def to_snake_case(s: str):
    """ Name Case to snake_case.  """
    return s.strip().lower().replace(" ", "_")


def get_results(
    subjects: Sequence[str] | None = None,
    years: Sequence[int] | None = None
) -> list[list[Results]]:  
    """ Gets model prediction results by running all models on
    VCAA exams from `subjects` and `years`. 

    Returns
    ------
    List of per-subject results for each model. 
    """
    if years is not None: 
        if not set(years).issubset(scope.YEARS):
            raise ValueError(f"{list(years)} is not a subset of {scope.YEARS}")
    else: 
        years = scope.YEARS

    scope_subj_std = [to_snake_case(subject) for subject in scope.SUBJECTS]
    if subjects is not None:
        subjects = [to_snake_case(subject) for subject in subjects]
        if not set(subjects).issubset(scope_subj_std):
            raise ValueError(f"{list(subjects)} is not a subset of {scope_subj_std}")
    else:
        subjects = scope_subj_std 

    # Extract material 
    for subj in subjects:
        if mc.main(argv = [subj, ",".join(map(str, years))]):
            raise Exception("Couldn't download material. ")
    
    # Run all models 
    models = [
        INSTRUCTOR('hkunlp/instructor-large'),
        SentenceTransformer("intfloat/e5-base-v2"),
        TfidfVectorizer()
    ]
    predictor_objs = [
        InstructPredictor,
        SentenceTransPredictor,
        TfIdfPredictor
    ]
    all_res: list[list[Results]] = []
    for model, predictor_obj in zip(models, predictor_objs):
        results: list[Results] = []
        for subj in subjects:
            # Load model here to re-use study design embeddings 
            sd_path = DATA_DIR / subj / "study_design" / f"{subj}_sd.docx"
            predictor = predictor_obj(sd_path, subj, model)
            for year in years:
                # find exam(s)
                exam_dir = DATA_DIR / subj / "past_exams"
                if mc.has_two_exams(subj):
                    exam_paths = [exam_dir / f"{year}_{i}.pdf" for i in [1, 2]]
                else: 
                    exam_paths = [exam_dir / f"{year}.pdf"]
                # Get results for this exam(s)
                for exam in exam_paths: 
                    res = predictor.run(exam)
                results.append(res)
            all_res.append(results)
    return all_res