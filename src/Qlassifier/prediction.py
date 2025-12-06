import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

from src.preprocessor.tree_preprocessor import TreePreprocessor, std_str
from src.Qlassifier.baseline import load_data

def get_mcq_predictions(
    qn_df: pd.DataFrame,
    sd_df: pd.DataFrame,
    labels: list[int],
    model: SentenceTransformer,
    subject: str = "",
    instruct: bool = False,
) -> tuple[pd.DataFrame, np.array, np.array]:
    """ Given a model, outputs prediction on """
    qn_input = qn_df["text"] 
    report_input = qn_df["comments"]
    sd_input = sd_df["label"].str.cat(sd_df["text"], sep="")

    if instruct:
        if not subject: 
            raise ValueError("If calling with instruct==True, must have subject non-empty.")

        sd_instruct = f"Represent this {std_str(subject)} topic:"
        report_instruct = f"Represent this {std_str(subject)} examination report comment:"
        qn_instruct = f"Represent this {std_str(subject)} question"

        sd_emb = model.encode(
            [[sd_instruct, input] for input in sd_input],
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        report_emb = model.encode(
            [[report_instruct, input] for input in report_input],
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        qn_emb = model.encode(
            [[qn_instruct, input] for input in qn_input],
            convert_to_tensor=True,
            normalize_embeddings=True
        )
    else: 
        sd_emb = model.encode(sd_input, convert_to_tensor=True, normalize_embeddings=True)
        report_emb = model.encode(report_input, convert_to_tensor=True, normalize_embeddings=True)
        qn_emb = model.encode(qn_input, convert_to_tensor=True, normalize_embeddings=True)

    cos_qn = util.cos_sim(qn_emb, sd_emb) # Embedding similarity b/w actual exam questions and study design
    best_idxs = [torch.argmax(sims).item() for sims in cos_qn]

    cos_rp = util.cos_sim(report_emb, sd_emb) # Embedding similarity b/w REPORT section on the question and study design
    best_rp_idxs = [torch.argmax(sims).item() for sims in cos_rp]

    qn_df["pred_topic"] = sd_df.loc[best_idxs, "label"].reset_index(drop=True)
    qn_df["pred_topic_idx"] = best_idxs # exam's prediction
    qn_df["report_pred"] = best_rp_idxs # report's prediction

    mcq_df = qn_df.loc[:(len(labels)-1), :].copy()
    mcq_df["true_topic_idx"] = labels
    mcq_df["true_topic"] = sd_df.loc[labels, "label"].reset_index(drop=True)
    return mcq_df, cos_qn, cos_rp


def prepare_dataframes(exam_path: str):
    subject = exam_path.parents[1].stem
    is_math = "math" in subject

    ex_root, report_df, sd_root = load_data(exam_path)
    sd_root = TreePreprocessor("study_design", remove_latex=True).preprocess(sd_root, subject=subject)
    ex_root = TreePreprocessor("past_exam", remove_latex=True).preprocess(ex_root)

    ex_level = ex_root.find_node_level(r"Question")
    sd_level = sd_root.find_node_level(r"Outcome \d") if "math" not in sd_root.subject_name \
                else (sd_root.find_node_level(r"Area of Study \d") + 1)
    
    ex_root.collapse(ex_level)
    if is_math:
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
    rp_df.drop_duplicates("question")
    rp_df = rp_df[["comments", "total_marks"]]
    
    ex_df = ex_df.loc[ex_df["label"].str.contains("Question", na=False), :].reset_index(drop=True)
    ex_df.drop_duplicates("label")

    qn_df = pd.concat([ex_df, rp_df], axis=1)
    return qn_df, sd_df