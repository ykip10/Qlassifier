from pathlib import Path

import pandas as pd 

from src.preprocessor.tree_preprocessor import TreePreprocessor
from src.parser.trees import Tree
from src.parser.parsers import AutoParser
from src.parser.report_processor import ReportProcessor


def load_data(exam_path: str) -> tuple[Tree, pd.DataFrame, Tree]:
    """ Loads all relevant data to the input exam_path. This includes:
      - The parsed exam document as a Tree
      - The parsed report as a pandas df
      - The parsed study design as a tree
    """
    exam_path = Path(exam_path)
    if not exam_path.exists():
        raise FileNotFoundError(f"No file found at {exam_path}.")
    
    exam = exam_path.stem
    subject_path = exam_path.parents[1]
    subject = subject_path.stem

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

    # Find study design
    sd_path = subject_path / "study_design" / f"{subject}_sd.docx"
    if not sd_path.exists():
        raise FileNotFoundError(f"Cannot find study design for {subject}")

    # load in data 
    ex_root = AutoParser(exam_path).parse()
    reports_df = ReportProcessor(report_path).parse_tables()
    sd_root = AutoParser(sd_path).parse()
    return ex_root, reports_df, sd_root


def prepare_dataframes(exam_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ Parses exams at the exam_path as well as the associated study designs and reports. 
    Performs some preprocessing then returns the exams as dataframes. 
    """
    subject = exam_path.parents[1].stem
    is_math = "math" in subject

    # Parsing
    ex_root, report_df, sd_root = load_data(exam_path)
    has_mcq = ex_root.has_mcq

    sd_root = TreePreprocessor("study_design", remove_latex=True).preprocess(sd_root, subject=subject)
    ex_root = TreePreprocessor("past_exam", remove_latex=True).preprocess(ex_root)

    ex_level = ex_root.find_node_level(r"Question")
    sd_level = sd_root.find_node_level(r"Outcome \d") if "math" not in sd_root.subject_name \
                else (sd_root.find_node_level(r"Area of Study \d") + 1)
    
    ex_root.collapse(ex_level)

    if is_math: # math study design slightly different formatting to the usual
        sd_root.collapse(sd_level, sep=": ")

    ex_df = ex_root.to_df(include_root=False)
    sd_df = sd_root.to_df(include_root=False, level=sd_level if is_math else 0)
    
    if not is_math: 
        sd_df = sd_df.loc[
            ~(sd_df["label"].str.contains(
                r"(Unit \d)|(Area of Study \d)|(Key Knowledge)|(Outcome \d)", case=False, na=False
            ) | sd_df["text"].str.contains("In this area of study"))
        ].reset_index(drop=True)

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
    
    ex_df = ex_df.loc[ex_df["label"].str.contains("Question", na=False), :].reset_index(drop=True)
    ex_df.drop_duplicates("label")

    qn_df = pd.concat([ex_df, rp_df], axis=1)
    return qn_df, sd_df