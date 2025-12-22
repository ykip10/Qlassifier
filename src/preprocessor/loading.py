""" High-level functions for loading data. Segmented into two types of functions:
    - loading data for model to directly absorb
    - loading study design mnaterial
"""
import re
from pathlib import Path

import pandas as pd 

from src.preprocessor.tree_preprocessor import TreePreprocessor
from src.parser.trees import Tree
from src.parser.parsers import AutoParser
from src.parser.report_processor import ReportProcessor
from src.extractor.material_collector import has_two_exams
from src.paths import DATA_DIR

#=========== LOADING EXAMS==========#
def load_exam(
    exam_path: str,
    subject: str,
    load_report: bool = False
) -> tuple[Tree, pd.DataFrame]:
    """ Loads all relevant data to the input `exam_path`. This includes:
      - The parsed exam document as a `Tree`
      - The parsed report as a pandas `DataFrame` (if `load_report == True`)
    """
    exam_path = Path(exam_path)
    if not exam_path.exists():
        raise FileNotFoundError(f"No file found at {exam_path}.")
    
    exam = exam_path.stem
    subject_path = DATA_DIR / subject
    # load in data 
    ex_root = AutoParser(exam_path).parse()
    reports_df = None
    if load_report: 
        # Find report
        report_dir = subject_path / "past_reports"
        # Check if report is a .pdf or .docx
        if (report_dir / f"{exam}.pdf").exists():
            ext = ".pdf"
        elif (report_dir / f"{exam}.docx").exists():
            ext =".docx"
        else: 
            raise FileNotFoundError(f"Cannot find associated report for exam at {exam_path}.")
        report_path = report_dir / f"{exam}{ext}"
        reports_df = ReportProcessor(report_path).parse_tables()

    return ex_root, reports_df


def prepare_exam_df(
    exam_path: str,
    subject: str,
    load_report: bool = False,
) -> pd.DataFrame:
    """ Parses exams at the `exam_path` as well as the associated report (if `load_report == True`). 
    Performs some preprocessing then returns the exam as dataframes.
    """
    # Parsing
    ex_root, report_df = load_exam(exam_path, subject, load_report)
    
    # Prepare exam df manually 
    has_mcq = ex_root.has_mcq
    ex_root = TreePreprocessor("past_exam", remove_latex=True).preprocess(ex_root)
    ex_level = ex_root.find_node_level(r"Question")
    ex_root.collapse(ex_level, label_sep="")
    # Convert to df
    ex_df = ex_root.to_df(include_root=False)
    # Remove labels which aren't Questions (section descriptions, formula sheets etc.)
    ex_df = ex_df.loc[
        ex_df["label"].str.contains("Question", na=False, case=False) |
        ex_df["label"].str.contains("Section", na=False, case=False)
    ].reset_index(drop=True)

    # Add section column by forward filling
    is_section = ex_df["label"].str.match(r"^Section\s+.?", na=False, case=False)
    is_question = ex_df["label"].str.match(r"^Question", na=False, case=False)
    ex_df["section"] = ex_df["label"].where(is_section).ffill()
    ex_df = ex_df[is_question].reset_index(drop=True)

    ex_df.drop_duplicates(subset=["label", "section"]) # Question section pairs are unique.
    
    # Finally, add question number column and sort by question number within sections
    def question_sort_key(q: str) -> tuple[int, str]:
        """ Sorting key for a question label. Sort by question number,
        then question part. Example; question 3 should come after question 2b 
        which comes after question 2a. 
        """
        m = re.match(r"(?i)Question\s+(\d+)([A-Z]*)", q)
        if not m:
            return (float("inf"), "")
        num = int(m.group(1))
        suffix = m.group(2) or ""
        return (num, suffix)
    
    # sorting
    ex_df["q_key"] = ex_df["label"].map(question_sort_key)
    ex_df = ex_df.sort_values(
        by=["section", "q_key"],
        kind="stable"
    ).reset_index(drop=True)

    ex_df["qn_num"] = ex_df["q_key"].apply(lambda x: x[0]) # qn number column
    ex_df = ex_df.drop(columns=["q_key"]) # dont need key anymore
    
    if report_df is not None:
        # Need to merge report data into the questions data
        rp_dfs = [df for df in report_df]
        # add marks column
        new_dfs = []
        for df in rp_dfs: 
            if "marks" in df.columns:
                df["total_marks"] = df.shape[1]-2
            else: 
                df["total_marks"] = 1
            new_dfs.append(df)
        rp_df = pd.concat(new_dfs).reset_index(drop=True)
        if has_mcq:
            rp_df.drop_duplicates("question")

        rp_df = rp_df[["comments", "total_marks"]] if "comments" in rp_df.columns else rp_df["total_marks"]
        qn_df = pd.concat([ex_df, rp_df], axis=1)
    else:
        qn_df = ex_df
    return qn_df


#========== LOADING STUDY DESIGNS ===========#
def load_sd(path: str) -> Tree:
    """ Loads the study design at `path`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find study design at {path}")
    sd_root = AutoParser(path).parse()
    return sd_root


def prepare_sd_df(
    path: str, 
    subject: str
) -> pd.DataFrame:
    """ Transforms a study design tree into a pandas DataFrame with two columns:
    `label`, and `text`. 
    """ 
    root = load_sd(path)

    two_exams = has_two_exams(subject)
    root = TreePreprocessor("study_design", remove_latex=two_exams).preprocess(root, subject=subject)

    df = root.to_df(
        include_root=False,
        level=(root.find_node_level(r"Area of Study \d") + 1) if two_exams else 0
    )
    if not two_exams: 
        # Again, this is because math s/d's are different since they're all merged into one
        # Remove all rows not directly related to the relevant subtopics
        df = df.loc[
            ~(df["label"].str.contains(
                r"(Unit \d)|(Area of Study \d)|(^Key Knowledge)|(^Outcome \d$)",
                case=False,
                na=False
            ) | df["text"].str.contains("In this area of study"))
        ].reset_index(drop=True)

    return df
