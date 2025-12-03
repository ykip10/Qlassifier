from typing import Literal
from pathlib import Path
import re
from copy import deepcopy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

from src.preprocessor.tree_preprocessor import TreePreprocessor
from src.parser.parsers import AutoParser
from src.parser.report_processor import ReportProcessor
from src.parser.trees import Tree

def run_tf_idf(
    exam_path: str,
    **vectorizerargs
) -> dict[str, str]:
    """ Runs TF-IDF on an exam and the subject's study design. Returns 
    dictionary mapping each question in the exam to a study design dot point,
    and the similarity scores for each dot point found in the parsed study design.
    """
    subject = Path(exam_path).parents[1].stem
    ex_root, reports_df, sd_root = load_data(exam_path)

    qns_labels, qns_desc = get_paragraphs(ex_root, subject=subject, doc_type="past_exam")
    topic_labels, topic_desc = get_paragraphs(sd_root, subject=subject, doc_type="study_design")

    # extract description from reports df
    ans_desc = []
    for table in reports_df:
        comments = list(table["comments"])
        ans_desc.extend(comments)
    ans_labels = qns_labels # report qn labels same as exam qn labels

    if len(ans_desc) != len(ans_labels):
        raise Exception("The number of questions extracted from report and exam differ. " \
                        "Likely an error in parsing error")
    

    vectorizer = TfidfVectorizer(**vectorizerargs)
    # Merge question text and report text to treat them as the same "document" for TF-IDF purposes. 
    qns_ans_merged = [(qn_desc + " " + ans_desc).strip() for \
                     qn_desc, ans_desc in zip(qns_desc, ans_desc)]
    # need to apply TF-IDF on the entire corpus, so we merge with the topics
    merged = topic_desc + qns_ans_merged

    # Run TF-IDF and store results
    tfidf = vectorizer.fit_transform(merged)
    topic_vecs = tfidf[:len(topic_desc)]
    ex_vecs = tfidf[len(topic_desc):]

    # Commpute cosine similarity then classify based on highest similarity
    similarity = cosine_similarity(ex_vecs, topic_vecs)
    assigned = np.argmax(similarity, axis=1)
    out_df = pd.DataFrame(
        data={
            "Question": qns_labels,
            "Predicted Topic": assigned, 
            **{f"{topic} (Similarity)": similarity[:, i] for i, topic in enumerate(topic_labels)}
        }
    )

    # add confidence column as similarity / sum(similarity)
    sim_cols = out_df.iloc[:, 2:]

    confidence_col = out_df.apply(
        lambda row: sim_cols.iloc[row.name, row["Predicted Topic"]] / sum(sim_cols.iloc[row.name, :]),
        axis=1
    )
    # Map to [0, 1]
    confidence_col = confidence_col / max(confidence_col)
    out_df["Confidence"] = confidence_col
    
    return out_df


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


def get_paragraphs(
    root: Tree,
    subject: str,
    doc_type: Literal["past_exam", "study_design"]
) -> tuple[list[str], list[str]]:
    """ Preprocesses all text in the tree.  Searches the tree for labels matching 
    level_re. Then, collapses the tree at this level, and returns the text associated with 
    all resulting nodes from this level as a tuple of lists; one for the text, one for the 
    associated labels
    """
    root = deepcopy(root)
    pre = TreePreprocessor(doc_type=doc_type, remove_latex=True) # initialise preprocessor
    is_math = "math" in subject

    # Find the level we need to collapse. 
    if doc_type == "past_exam":
        level_re = r"Question"
    else: 
        level_re = r"Outcome \d" if not is_math else r"Area of Study \d"
    
    tg_level = root.find_node_level(level_re)
    if doc_type == "study_design" and not is_math:
        tg_level += 1
        
    # preprocess then collapse
    root = pre.preprocess(root=root, subject=subject)
    root.collapse(level=tg_level, concat_label=doc_type!="past_exam")

    # Get the text and labels as a list for use in tf-idf
    nodes = root.get_nodes_at_level(tg_level)
    if doc_type == "past_exam":
        labels = [normalise_question_label(qn.label) for qn in nodes]
        desc = [qn.text for qn in nodes]
    else: 
        labels = [node.label for node in nodes]
        desc = [node.text for node in nodes]

    return labels, desc 


def normalise_question_label(s: str) -> str:
    """ Normalises question labels which become like Question 1.\tc\ti after collpasing
    to be of the form Question 1ci. instead. 
    """
    # Remove all tabs
    pref = re.search(r"Question \d{1,2}", s).group()
    suff = "".join(c for c in s.replace(pref, "") if c.isalnum())
    return pref + suff + "."