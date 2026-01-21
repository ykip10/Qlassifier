""" Builds the dataset to be labelled.  Goes through every subject and year in scope
 and compiles a semi-labelled dataset
"""
from typing import Sequence
from collections import defaultdict as dd

from sklearn.feature_extraction.text import TfidfVectorizer
from InstructorEmbedding import INSTRUCTOR
from sentence_transformers import SentenceTransformer
import pandas as pd

from src.Qlassifier.results import Results
from src.Qlassifier.prediction import (SentenceTransPredictor,
                                       InstructPredictor,\
                                        TfIdfPredictor)
from src.extractor import materialCollector as mc
from src.paths import DATA_DIR
import scope


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
    model_aliases = [
        "instructor",
        "sentence-transformer",
        "tf-idf"
    ] 
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

    all_res: dict[
        str,
        dict[str, list[Results]]
    ] = []
    for model_alias, model, predictor_obj in zip(model_aliases, models, predictor_objs):
        results: dict[str, Results] = dd(list)
        for subj in subjects:
            # Load model here to re-use study design embeddings 
            sd_path = DATA_DIR / subj / "study_design" / f"{subj}_sd.docx"
            predictor = predictor_obj(sd_path, subj, model)
            subj_results = []
            for year in years:
                # find exam(s)
                exam_dir = DATA_DIR / subj / "past_exams"
                # If the subject has two exams per year, we need to process multiple exams
                exam_names = [str(year)] if not mc.has_two_exams(subj) else \
                             [f"{year}_{i}" for i in [1, 2]]
                exam_paths = [exam_dir / f"{exam}.pdf" for exam in exam_names]
                # Get results for this years exam(s), then merge into one dataframe
                year_results = [predictor.run(exam) for exam in exam_paths]
                for i, res in enumerate(year_results):
                    res.add_pred_column("exam_name", exam_names[i])
                year_results_merged = pd.concat(year_results, ignore_index=True)     
                subj_results.append(year_results_merged)
            # Finally, merge all exams for this subject
            subj_results_merged = pd.concat(subj_results, ignore_index=True)
            results[subj] = subj_results_merged
        all_res[model_alias] = results
    return all_res
