from typing import Union

import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from InstructorEmbedding import INSTRUCTOR

from src.preprocessor.tree_preprocessor import std_str

def get_predictions(
    qn_df: pd.DataFrame,
    sd_df: pd.DataFrame,
    labels: list[int],
    model: Union[SentenceTransformer, INSTRUCTOR],
    subject: str = "",
    instruct: bool = False,
) -> tuple[pd.DataFrame, np.array, np.array]:
    """ Given a model which is either a Sentence Transformer or extractor, 
    outputs prediction for a topic classification for each question in qn_df.

    qn_df   : Dataframe containing atleast questions and examiner comments.
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
    pred_df = qn_df.loc[:(len(labels)-1)].copy()
    qn_input = pred_df["text"] 
    report_input = pred_df["comments"]
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

    cos_qn = util.cos_sim(qn_emb, sd_emb) # Embedding similarity b/w actual exam questions and study design
    best_idxs = [torch.argmax(sims).item() for sims in cos_qn]

    cos_rp = util.cos_sim(report_emb, sd_emb) # Embedding similarity b/w REPORT section on the question and study design
    best_rp_idxs = [torch.argmax(sims).item() for sims in cos_rp]

    # Add predictions column 
    pred_df["pred_topic"] = sd_df.loc[best_idxs, "label"].reset_index(drop=True)
    pred_df["pred_topic_idx"] = best_idxs # exams prediction
    pred_df["report_pred"] = best_rp_idxs # reports prediction

    pred_df["true_topic_idx"] = labels 
    pred_df["true_topic"] = sd_df.loc[labels, "label"].reset_index(drop=True)

    # Add confidence of question texts' prediction
    confidence_col = np.array([(sims[pred_idx] / sum(sims)).cpu() for pred_idx, sims in zip(best_idxs, cos_qn)])
    # Map to [0, 1]
    confidence_col = (confidence_col - min(confidence_col)) / (max(confidence_col) - min(confidence_col))
    pred_df["confidence"] = confidence_col
    return pred_df, cos_qn, cos_rp
