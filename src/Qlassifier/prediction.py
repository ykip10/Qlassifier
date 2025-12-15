""" This model contains all logic relating to running the model on an input exam. """

from typing import Any, Sequence
from abc import ABC, abstractmethod
from pathlib import Path
import re

import torch
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from InstructorEmbedding import INSTRUCTOR

from src.preprocessor.tree_preprocessor import std_str
from src.preprocessor.loading import load_sd, prepare_sd_df, prepare_dataframes
from src.Qlassifier.results import Results

class Predictor(ABC):
    @abstractmethod
    def classify(
        self,
        text: str,
        return_cos: bool = False
    ) -> str | Sequence[float]:
        pass
    
    @abstractmethod
    def run(self, exam_path: str) -> Results:
        pass


class InstructPredictor(Predictor):
    def __init__(self, sd_path: str, subject: str):
        self.model = INSTRUCTOR('hkunlp/instructor-large')
        self.subject = subject
        self.sd_df = prepare_sd_df(load_sd(sd_path), subject=subject)
        self.sd_emb = self._get_sd_emb()

    def classify(
        self,
        questions: Sequence[str],
        return_cos: bool = False
    ) -> str | Sequence[float]:
        """ Classifies the input sequence of text into one of the study design topics each. 
        If `return_cos == "True"` returns the cosine similarity matrix. Else,
        just returns the study design topics with highest cosine similarities.  
        """
        subject = self.subject
        if "math" in self.subject:
            # Instructor doesn't really understand useless quantifiers like "Specialist" in 
            # "Specialist Mathematics"
            subject = "Mathematics" 
        qn_instruct = f"Represent this {std_str(subject)} question for semantic search: "
        qn_emb = self.model.encode(
            [[qn_instruct, question] for question in questions],
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        cos = util.cos_sim(qn_emb, self.sd_emb)
        if return_cos: 
            return cos
        best_idxs = torch.argmax(cos, dim=1).tolist()
        return self.sd_df.loc[best_idxs, "label"]

    def run(self, exam_path: str, include_report: bool = False):
        pred_df = prepare_dataframes(exam_path, subject=self.subject, load_report=include_report)[0]
        qn_input = pred_df["text"]
        cos = self.classify(qn_input, return_cos=True)
        best_idxs = torch.argmax(cos, dim=1).tolist()

        # Append our predictions to the questions DataFrame
        pred_df["pred_topic"] = self.sd_df.loc[best_idxs, "label"].reset_index(drop=True)
        pred_df["pred_topic_idx"] = best_idxs

        # Add confidence of question texts' prediction
        conf_col = torch.max(cos, axis=1).values / cos.sum(dim=1)
        # Map to [0, 1]
        conf_col = ((conf_col - min(conf_col)) / (max(conf_col) - min(conf_col))).tolist()
        pred_df["confidence"] = conf_col
        if include_report:
            report_input = pred_df["comments"] if "comments" in pred_df.columns else [""]*len(qn_input)
            cos_rp = self.classify(report_input, return_cos=True)
            best_rp_idxs = torch.argmax(cos_rp, dim=1).tolist()
            pred_df["report_pred"] = best_rp_idxs
        
        results = Results(pred_df, cos, self.sd_df["label"])
        return results

    def _get_sd_emb(self):
        """ Gets study design embeddings. """
        df = self.sd_df
        input = df["label"].str.cat(df["text"], sep="")
        instruct = f"Represent this {std_str(self.subject)} topic for semantic search: "
        sd_emb = self.model.encode(
            [[instruct, i] for i in input],
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        return sd_emb


class TfIdfPredictor(Predictor):
    """ For tf-idf, we need some sort of common corpus to use in classifying arbitrary text.
    We will use historic exam data to build this corpus.  
    """
    def __init__(
        self,
        sd_path: str,
        subject: str, 
        last_year: int | None = None,
        **vectorizerargs
    ):  
        """ Initialises TfIdfPredictor. 

        `sd_path`     : Path pointing to relevant study design. 
        `subject`     : The subject name. 
        `corpus_years`: The earliest year we should look for past exams in 
                      the building of the tf-idf corpus.
        """
        self.vectorizer = TfidfVectorizer(**vectorizerargs)
        self.subject = subject
        self.sd_path = sd_path
        self.last_year = last_year if last_year is not None else 0
        # Need to extract a corpus.
        self.corpus = self._load_corpus()
    
    def _load_corpus(self):
        sd_df = prepare_sd_df(load_sd(self.sd_path), subject=self.subject)
        sd_in = sd_df["label"]
        exam_dir = Path(self.sd_path).parents[0] / "past_exams"
        for exam in exam_dir.iterdir():
            exam_name = str(exam)
            match = re.search(r"\d{4}", exam_name) # extract year
            year = int(match.group())
            if year < self.last_year:
                continue

        # To finish implementing


def run_combined(
    exam_path: str,
    subject: str, 
    tf_idf_weight: float = 0.5,
    *,
    model: INSTRUCTOR | None = None,
    **vectorizerargs: dict[str, Any],
) -> Results:
    """ Runs both an INSTRUCTOR model and tf-idf model, then merges predictions. 
    It finds the normalised cosine matrices for each model then computes the element-wise
    vector sum, where the tf_idf predictions are weighed by `tf_idf_weight`, before
    renormalising.
    """
    inst_results = run_instructor(
        exam_path,
        subject,
        model=model
    )
    tf_results = run_tf_idf(
        exam_path,
        subject,
        include_report=False,
        **vectorizerargs
    )
    cos_inst = inst_results.cos
    cos_tf = tf_results.cos
    sd_labels = inst_results.sd_labels
    # Compute weighted sum of similarities then re-normalise
    cos_inst = cos_inst.cpu().numpy()
    cos = cos_inst + tf_idf_weight * cos_tf

    # min-max normalise (within predictions)
    row_maxes = cos.max(axis=1, keepdims=True)
    row_mins  = cos.min(axis=1, keepdims=True)
    cos = (cos - row_mins) / (row_maxes - row_mins)

    # update prediction df
    pred_df = inst_results.pred_df.copy()
    pred_df["pred_topic_idx"] = cos.argmax(axis=1)

    results = Results(pred_df, cos, sd_labels)
    return results


def run_instructor(
    exam_path: str,
    subject: str, 
    model: INSTRUCTOR | None = None
) -> Results:
    """ Runs the instructor model on the exam at exam_path. Returns Question labels 
    and the cosine similarity matrix.
    """
    if model is None: 
        # Default model 
        model = INSTRUCTOR('hkunlp/instructor-large')

    qn_df, sd_df = prepare_dataframes(exam_path, subject)
    results = get_predictions(qn_df, sd_df, model, subject=subject)
    return results


def run_tf_idf(
    exam_path: str,
    subject: str, 
    labels: list[str] | None = None,
    include_report: bool = True,
    **vectorizerargs
) -> Results:
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

    results = Results(pred_df, cos, list(sd_df["label"]))
    return results


def get_predictions(
    qn_df: pd.DataFrame,
    sd_df: pd.DataFrame,
    model: SentenceTransformer | INSTRUCTOR,
    subject: str,
    labels: list[int] | None = None,
) -> Results:
    """ Given a transformers model which is either a Sentence Transformer or extractor, 
    outputs prediction for a topic classification for each question in qn_df.

    qn_df   : Dataframe containing at least questions and examiner comments.
    sd_df   : Dataframe containing study design dot points which the questions in qn_df
              should be mapped to
    labels  : Ground truth topic labels for each question
    model   : Any SentenceTransformer or INSTRUCTOR model. 
    subject : The subject the examination is assessing. Only needed if instruct == True
              (in which case it is required)

    Returns the dataframe containing predictions as well as cosine similarity matrices for 
    both question-driven predictions and examiner comments-driven predictions. 
    """
    instruct = isinstance(model, INSTRUCTOR) 

    pred_df = qn_df.loc[:(len(labels)-1)].copy() if labels is not None else qn_df.copy()
    qn_input = pred_df["text"]
    # we don't always have a comments column
    report_input = pred_df["comments"] if "comments" in pred_df.columns else [""]*len(qn_input)
    sd_input = sd_df["label"].str.cat(sd_df["text"], sep="")
    # Find embeddings
    if instruct:
        if not subject: 
            raise ValueError("If calling with instructor model, must have subject non-empty.")
        if "math" in subject:
            # Instructor doesn't really understand "Specialist Mathematics"
            subject = "Mathematics" 

        sd_instruct = f"Represent this {std_str(subject)} topic for semantic search: "
        report_instruct = f"Represent this {std_str(subject)} examination report comment: "
        qn_instruct = f"Represent this {std_str(subject)} question for semantic search: "

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

    if labels is not None: 
        pred_df["true_topic_idx"] = labels 
        pred_df["true_topic"] = sd_df.loc[labels, "label"].reset_index(drop=True)

    # Add confidence of question texts' prediction
    conf_col = torch.max(cos_qn, axis=1).values / cos_qn.sum(dim=1)
    # Map to [0, 1]
    conf_col = ((conf_col - min(conf_col)) / (max(conf_col) - min(conf_col))).tolist()
    pred_df["confidence"] = conf_col

    results = Results(pred_df, cos_qn, sd_df["label"], correct_topics=labels if labels is not None else None)
    return results
