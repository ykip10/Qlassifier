""" This model contains all logic relating to running the model on an input exam. """

from typing import Any

import torch
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from InstructorEmbedding import INSTRUCTOR

from src.preprocessor.tree_preprocessor import std_str
from src.Qlassifier.utils import prepare_dataframes

def run_combined(
    exam_path: str,
    subject: str, 
    tf_idf_weight: float = 0.5,
    *,
    model: INSTRUCTOR | None = None,
    **vectorizerargs: dict[str, Any],
) -> tuple[
    pd.DataFrame, 
    list[str], 
    torch.FloatTensor
]:
    """ Runs both an INSTRUCTOR model and tf-idf model, then merges predictions. 
    It finds the normalised cosine matrices for each model then computes the element-wise
    vector sum, where the tf_idf predictions are weighed by `tf_idf_weight`, before
    renormalising.
    """
    qn_labels, cos_inst = run_instructor(
        exam_path,
        subject,
        model=model
    )
    sd_labels, cos_tf = run_tf_idf(
        exam_path,
        subject,
        include_report=False,
        **vectorizerargs
    )[1:]
    # Compute weighted sum of similarities then re-normalise
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cos_tf = torch.from_numpy(cos_tf).float().to(device)
    cos = cos_inst + tf_idf_weight * cos_tf

    row_maxes = cos.max(dim=1, keepdim=True).values
    row_mins = cos.min(dim=1, keepdim=True).values
    cos = (cos - row_mins) / (row_maxes - row_mins)
    return qn_labels, sd_labels, cos


def run_instructor(
    exam_path: str,
    subject: str, 
    model: INSTRUCTOR | None = None
) -> tuple[
    list[str],
    torch.FloatTensor
]:
    """ Runs the instructor model on the exam at exam_path. Returns top 3 topics 
    for each question, as well as the question labels. 
    """
    if model is None: 
        # Default model 
        model = INSTRUCTOR('hkunlp/instructor-large')

    qn_df, sd_df = prepare_dataframes(exam_path, subject)
    cos_qn = get_predictions(qn_df, sd_df, model, subject=subject, instruct=True)[1]
    return list(qn_df["label"]), cos_qn


def run_tf_idf(
    exam_path: str,
    subject: str, 
    labels: list[str] | None = None,
    include_report: bool = True,
    **vectorizerargs
) -> tuple[pd.DataFrame, pd.Series, torch.FloatTensor]:
    """ Runs TF-IDF on an exam and the subject's study design.
    Returns:
        - dataframe containing predictions (optionally with correct labels, if provided)
        - study design topic labels
        - cosine similarity matrix
    """ 
    qn_df, sd_df = prepare_dataframes(exam_path, subject)

    ans_desc = []
    if not include_report:
        if "comments" in qn_df.columns:          
            qn_df.drop(columns=["comments"])
    else: 
        ans_desc = qn_df["comments"]
    qns_desc = qn_df["text"]
    topic_desc = list(sd_df["text"])
    
    vectorizer = TfidfVectorizer(**vectorizerargs)
    # Merge question text and report text to treat them as the same "document" for TF-IDF purposes. 
    qns_ans_merged = [(qn_desc + " " + ans_desc).strip() for \
                     qn_desc, ans_desc in zip(qns_desc, ans_desc)] if include_report else \
                     [qn_desc.strip() for qn_desc in qns_desc]
    # need to apply TF-IDF on the entire corpus, so we merge with the topics
    merged = topic_desc + qns_ans_merged

    # Run TF-IDF and store results
    tfidf = vectorizer.fit_transform(merged)
    topic_vecs = tfidf[:len(topic_desc)]
    ex_vecs = tfidf[len(topic_desc):]

    # Commpute cosine similarity then classify based on highest similarity
    cos = cosine_similarity(ex_vecs, topic_vecs)
    best_idxs = np.argmax(cos, axis=1)

    # add confidence column as similarity / sum(similarity)
    conf_col = np.max(cos, axis=1)/ (np.sum(cos, axis=1)+1e-12)

    # Map to [0, 1]
    conf_col = ((conf_col - min(conf_col)) / (max(conf_col) - min(conf_col))).tolist()

    pred_df = qn_df.copy()
    pred_df["pred_topic"] = sd_df.loc[best_idxs, "label"].reset_index(drop=True)
    pred_df["pred_topic_idx"] = best_idxs
    if labels is not None: 
        pred_df["true_topic_idx"] = labels 
        pred_df["true_topic"] = sd_df.loc[labels, "label"].reset_index(drop=True)
    pred_df["confidence"] = conf_col
    return pred_df, sd_df["label"], cos


def get_predictions(
    qn_df: pd.DataFrame,
    sd_df: pd.DataFrame,
    model: SentenceTransformer | INSTRUCTOR,
    labels: list[int] = [],
    subject: str = "",
    instruct: bool = False,
) -> tuple[pd.DataFrame, torch.FloatTensor, torch.FloatTensor]:
    """ Given a transformers model which is either a Sentence Transformer or extractor, 
    outputs prediction for a topic classification for each question in qn_df.

    qn_df   : Dataframe containing at least questions and examiner comments.
    sd_df   : Dataframe containing study design dot points which the questions in qn_df
              should be mapped to
    labels  : Ground truth topic labels for each question
    model   : Any SentenceTransformer or INSTRUCTOR model. 
    subject : The subject the examination is assessing. Only needed if instruct == True
              (in which case it is required)
    instruct: Whether or not we should input instruction into the model (INSTRUCTOR model only). 

    Returns the dataframe containing predictions as well as cosine similarity matrices for 
    both question-driven predictions and examiner comments-driven predictions. 
    """
    pred_df = qn_df.loc[:(len(labels)-1)].copy() if labels else qn_df.copy()
    qn_input = pred_df["text"] 
    report_input = pred_df["comments"] if "comments" in pred_df.columns else [""]*len(qn_input)
    sd_input = sd_df["label"].str.cat(sd_df["text"], sep="")
    # Find embeddings
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

    # Embedding similarity b/w actual exam questions and study design
    cos_qn = util.cos_sim(qn_emb, sd_emb) 
    best_idxs = torch.argmax(cos_qn, dim=1).tolist()

    # Embedding similarity b/w REPORT section on the question and study design
    cos_rp = util.cos_sim(report_emb, sd_emb) 
    best_rp_idxs = torch.argmax(cos_rp, dim=1).tolist()

    # Add predictions column 
    pred_df["pred_topic"] = sd_df.loc[best_idxs, "label"].reset_index(drop=True)
    pred_df["pred_topic_idx"] = best_idxs # exams prediction
    pred_df["report_pred"] = best_rp_idxs # reports prediction

    if labels: 
        pred_df["true_topic_idx"] = labels 
        pred_df["true_topic"] = sd_df.loc[labels, "label"].reset_index(drop=True)

    # Add confidence of question texts' prediction
    conf_col = torch.max(cos_qn, axis=1).values / cos_qn.sum(dim=1)
    # Map to [0, 1]
    conf_col = ((conf_col - min(conf_col)) / (max(conf_col) - min(conf_col))).tolist()
    pred_df["confidence"] = conf_col
    return pred_df, cos_qn, cos_rp


