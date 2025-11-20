from __future__ import annotations 
from typing import TYPE_CHECKING

from . import material_collector as mc, visualise_parsing as mp
from paths import DATA_DIR

if TYPE_CHECKING:
    from trees import Tree


def process_exams(
    subjects: list[str],
    years: list[int],
    reports: bool = True
) -> tuple[
    dict[str, list[Tree]], 
    dict[str, list[Tree]]
]:
    # Check if we already have some exams downloaded
    # Download PDFs then process them
    all_exams = {}
    all_reports = {} if reports else None
    for subject in subjects:
        print(f"======= {subject} =======")
        subject_dir = DATA_DIR / f"{subject.lower().replace(' ', '_')}"
        exams_dir = subject_dir / "past_exams"
        reports_dir = subject_dir / "past_reports"

        # Downloading
        print("Downloading exams", " and reports..." if reports else "...")
        if not mc.extract_exams(subject, years, reports):
            print("Download failed. Exiting.")
            return None
        else:
            print("Success!")
        
        # Processing
        print("Processing exams...")
        subject_exams = []
        subject_reports = []
        for file_name in exams_dir.iterdir():
            # process exams
            exam = mp.process_pdf(exams_dir / file_name)
            if exam is None:
                print("Error processing exam. Exiting.")
                return None
            subject_exams.append(exam)

            # if we need to process reports, process them as well 
            if reports:
                report = mp.extract_headings(reports_dir / file_name)
                if report is None:
                    print("Error processing report. Exiting.")
                    return None
                subject_reports.append(report)
        
        all_exams[subject] = subject_exams
        if reports:
            all_reports[subject] = subject_reports
        print("Success!")
    
    return all_exams, all_reports


def process_sds(subjects: list[str]):
    pass