""" High-level functions for loading data. Segmented into two types of functions:
    - loading data for model to directly absorb
    - loading study design mnaterial
"""
from pathlib import Path

import pandas as pd 

from src.preprocessor.tree_preprocessor import TreePreprocessor
from src.parser.trees import Tree
from src.parser.parsers import AutoParser
from src.parser.report_processor import ReportProcessor
from src.paths import DATA_DIR

def load_data(
    exam_path: str,
    subject: str,
    load_report: bool = False
) -> tuple[Tree, pd.DataFrame, Tree]:
    """ Loads all relevant data to the input exam_path. This includes:
      - The parsed exam document as a Tree
      - The parsed report as a pandas df
      - The parsed study design as a tree
    """
    exam_path = Path(exam_path)
    if not exam_path.exists():
        raise FileNotFoundError(f"No file found at {exam_path}.")
    
    exam = exam_path.stem
    subject_path = DATA_DIR / subject

    # Find study design
    sd_path = subject_path / "study_design" / f"{subject}_sd.docx"
    sd_root = load_sd(sd_path)

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

    return ex_root, reports_df, sd_root


def prepare_dataframes(
    exam_path: str,
    subject: str,
    load_report: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ Parses exams at the exam_path as well as the associated study designs and reports. 
    Performs some preprocessing then returns the exams as dataframes. 
    """
    # Parsing
    ex_root, report_df, sd_root = load_data(exam_path, subject, load_report)
    sd_df = prepare_sd_df(sd_root, subject) # prepare sd_df 
    
    # Prepare exam df manually 
    has_mcq = ex_root.has_mcq
    ex_root = TreePreprocessor("past_exam", remove_latex=True).preprocess(ex_root)
    ex_level = ex_root.find_node_level(r"Question")
    ex_root.collapse(ex_level)
    # Convert to df
    ex_df = ex_root.to_df(include_root=False)
    # Remove labels which aren't Questions (section descriptions, formula sheets etc.)
    ex_df = ex_df.loc[ex_df["label"].str.contains("Question", na=False), :].reset_index(drop=True)
    ex_df.drop_duplicates("label") # Questions are unique. 

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
    return qn_df, sd_df


#========== LOADING STUDY DESIGNS IN ISOLATION ===========#
def load_sd(path: str):
    """ Loads the study design at `path`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find study design at {path}")
    sd_root = AutoParser(path).parse()
    return sd_root


def prepare_sd_df(
    root: Tree,
    subject: str
) -> pd.DataFrame:
    """ Transforms a study design tree into a pandas DataFrame with two columns:
    label, and text. 
    """ 
    is_math = "math" in subject
    root = TreePreprocessor("study_design", remove_latex=True).preprocess(root, subject=subject)
    # +1 for math s/d's since we want to collapse exactly one indentation after 
    # Area of study headers 
    level = root.find_node_level(r"Outcome \d") if not is_math \
                else (root.find_node_level(r"Area of Study \d") + 1)
    if is_math:
        # For math study designs, need to collapse to get 
        # full label context
        root.collapse(level, sep=": ")
    else:
        # For non-math study designs, dot points are under outcomes. 
        root.filter_tree(r"Outcome \d")

    # convert Tree to pandas dataframe
    df = root.to_df(include_root=False, level=level if is_math else 0)
    if not is_math: 
        # Again, this is because math s/d's are different since they're all merged into one
        # Remove all rows not directly related to the relevant subtopics
        df = df.loc[
            ~(df["label"].str.contains(
                r"(Unit \d)|(Area of Study \d)|(Key Knowledge)|(Outcome \d)",
                case=False,
                na=False
            ) | df["text"].str.contains("In this area of study"))
        ].reset_index(drop=True)

    return df
