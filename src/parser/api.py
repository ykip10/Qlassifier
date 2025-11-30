""" High level functions for collecting and parsing documents. """
from __future__ import annotations 
from typing import TYPE_CHECKING, Literal, Union

import loader.material_collector as mc
from parser.report_processor import ReportProcessor
from parser.parsers import AutoParser
from paths import DATA_DIR


if TYPE_CHECKING:
    import pandas as pd
    from parser.trees import Tree


def process_material(
    subjects: list[str],
    years: list[int],
    type: Literal["exam", "report", "study_design"]
) -> dict[
    str, 
    Union[
        Tree,
        dict[str, Union[Tree, list[pd.DataFrame]]]
    ]
]:  
    """ Downloads material requested by type, then processes them. Returns 
    a dictionary mapping each subject to the processed output. """
    
    # Sort out function definitions based on the type of document 
    def parse(path: str):
        parser = AutoParser(path).parser
        return parser.parse()
    
    if type == "exam":
        def extractor(subject, years):
            return mc.vcaa_extrapct_exam_materials(subject, years, reports=False)
        folder_name = mc.EXAM_DIR_NAME

    elif type == "report":
        def extractor(subject, years): 
            return mc.vcaa_extract_exam_materials(subject, years, exams=False)
        def parse(path: str):
            parser = ReportProcessor(path)
            return parser.parse_tables()
        folder_name = mc.RP_DIR_NAME

    elif type == "study_design":
        def extractor(subject, years):
            return mc.extract_sds(subject)
        folder_name = mc.SD_DIR_NAME

    else: 
        raise ValueError("Argument 'type' must be one of  ['exam', 'report, 'study_design']")
    
    # Execute main logic 
    all_docs = {}
    for subject in subjects:
        subject_dir = DATA_DIR / f"{subject.lower().replace(' ', '_')}"
        dir_name = subject_dir / folder_name

        # Downloading
        if not extractor(subject, years):
            print("Report download failed. Exiting.")
            return None
        
        # Processing
        subject_docs = {}
        for file_name in dir_name.iterdir():
            if file_name.suffix not in [".docx", ".pdf"] or not \
                any(str(year) in str(file_name) for year in years):
                continue

            result = parse(file_name)
            if result is None or not result: 
                print(f"Error parsing document {file_name}.")
                return None
            subject_docs[file_name.stem] = result

        all_docs[subject] = subject_docs
    
    return all_docs
