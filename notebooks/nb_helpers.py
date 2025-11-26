""" Module containing helper functions for notebook analyses. 
"""
import pandas as pd

def mcq_merged(
    all_reports: dict[str, list[pd.DataFrame]], 
    years: list[int],
) -> dict[str, pd.DataFrame]:
    """ Returns a dictionary mapping each subject to one dataframe consisting
    of all multiple choice tables merged over the years indicated by years.
    """
    all_merged = {}
    subjects = all_reports.keys
    for subject in subjects:
        subject_reports = all_reports[subject]
        report_types = subject_reports.keys

        dfs_to_merge = [report[0] for report in report_types if any(year in report for year in years)]
        merged = pd.concat(dfs_to_merge, axis=1)
        all_merged[subject] = merged
    return all_merged



def set_correct_answer(all_reports) -> pd.DataFrame:
    """ For MCQ dfs without a correct answer field, this function's purpose 
    is to set create a proxy for the correct answer by looking at the mode 
    response selection among students. It returns the df with this added field.
    """
    all_reports["chemistry"]