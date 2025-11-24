from __future__ import annotations 
from typing import TYPE_CHECKING

import src.Qlassifier.material_collector as mc
from src.Qlassifier.report_processor import ReportProcessor
from src.Qlassifier.parsers import WordParser, PDFParser
from src.Qlassifier.paths import DATA_DIR


if TYPE_CHECKING:
    from trees import Tree


def process_exams(
    subjects: list[str],
    years: list[int]
) -> dict[str, list[Tree]]:
    all_exams = {}
    for subject in subjects:
        print(f"======= {subject} =======")
        subject_dir = DATA_DIR / f"{subject.lower().replace(' ', '_')}"
        exams_dir = subject_dir / "past_exams"

        # Downloading
        print("Downloading exams...")
        if not mc.vcaa_extract_exam_materials(subject, years, reports=False):
            print("Download failed. Exiting.")
            return None
        else:
            print("Success!")
        
        # Processing
        print("Processing exams...")
        subject_exams = []
        for file_name in exams_dir.iterdir():
            # check if it's pdf or word
            parser = PDFParser(file_name) if file_name.suffix == ".pdf" else WordParser(file_name)
            # process exams
            exam = parser.split_headings()
            if exam is None:
                print("Error processing exam. Exiting.")
                return None
            subject_exams.append(exam)
        
        all_exams[subject] = subject_exams
        print("Success!")
    
    return all_exams


def process_reports(
    subjects: list[str],
    years: list[int]
) -> dict[str, "list[pd.DataFrame]"]:
    all_reports = {}
    for subject in subjects:
        subject_dir = DATA_DIR / f"{subject.lower().replace(' ', '_')}"
        reports_dir = subject_dir / "past_exams"

        # Downloading
        if not mc.vcaa_extract_exam_materials(subject, years, exams=False):
            print("Report download failed. Exiting.")
            return None
        else:
            print("Success!")
        
        # Processing
        subject_reports = []
        for file_name in reports_dir.iterdir():
            # check if it's pdf or word
            processor = ReportProcessor(file_name)
            # process exams
            report = processor.parse_tables()
            if report is None:
                print("Error processing report. Exiting.")
                return None
            subject_reports.append(report)
        
        all_reports[subject] = subject_reports
    
    return all_reports
    
    

def process_sds(subjects: list[str]):
    pass