""" This module contains all logic relating to running the model on an input exam. """

from typing import Sequence
from abc import ABC, abstractmethod
from pathlib import Path

import torch
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from InstructorEmbedding import INSTRUCTOR

from src.preprocessor.tree_preprocessor import std_str
from src.preprocessor.loading import prepare_sd_df, prepare_exam_df
from src.Qlassifier.results import Results


class TransformerPredictor(ABC):
    def __init__(self, sd_path: str, subject: str):
        self.subject = subject
        self.sd_df = prepare_sd_df(sd_path, subject=subject)
        self.sd_emb = self._get_sd_emb()

    @abstractmethod
    def embed(self, text: Sequence[str]) -> torch.Tensor:
        """ Embeds input text. """
        pass

    @abstractmethod
    def _get_sd_emb(self) -> torch.Tensor:
        """ Get study design embeddings. """
        pass

    def classify(
        self,
        text: Sequence[str],
        return_cos: bool = False
    ) -> str | Sequence[float]:
        """ Classifies the input sequence of text into one of the study design topics each. 
        If `return_cos == "True"` returns the cosine similarity matrix. Else,
        just returns the study design topics with highest cosine similarities.  
        """
        qn_emb = self.embed(text)
        cos = util.cos_sim(qn_emb, self.sd_emb)
        if return_cos: 
            return cos
        best_idxs = torch.argmax(cos, dim=1).tolist()
        return list(self.sd_df.loc[best_idxs, "label"])
    
    def run(
        self,
        exam: str | Path | pd.DataFrame,
        include_report: bool = False
    ) -> Results:
        """ Runs the predictor on an exam, and returns the result      
        as a `Results` object. 

        `exam`          : Either a string/path object pointing to an exam, or a pandas dataframe
                          Containing the questions of an exam and their associated text. 
        `include_report`: Whether or not to draw on the examiner's report to assist with classification.
                          Can only be `True` if the exam points to a path. 
        """
        if isinstance(exam, str) or isinstance(exam, Path):
            pred_df = prepare_exam_df(exam, subject=self.subject, load_report=include_report)
        elif isinstance(exam, pd.DataFrame):
            if include_report:
                raise ValueError("Cannot include report if exam is a pandas DataFrame.")
            pred_df = exam.copy()
        else: 
            raise TypeError("exam must be a str, Path or pandas DataFrame.")

        qn_input = pred_df["text"]
        cos = self.classify(qn_input, return_cos=True) # cosine sim matrix
        # Find max of cosine sim matrix for each question (this is our prediction)
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
        
        results = Results(pred_df, cos, self.subject, self.sd_df["label"])
        return results
    

class InstructPredictor(TransformerPredictor):
    def __init__(
        self,
        sd_path: str,
        subject: str,
        model: INSTRUCTOR | None = None
    ):
        self.model = model if model is not None else INSTRUCTOR('hkunlp/instructor-large')
        super().__init__(sd_path, subject)
        
    def embed(
        self,
        questions: Sequence[str]
    ) -> torch.Tensor:
        """ Returns `questions` representation in the models' embedding space. 
        Automatically inputs an instruction relevant to the initialised subjects' domain. 
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
        return qn_emb

    def _get_sd_emb(self):
        """ Gets study design embeddings. """
        df = self.sd_df
        input = df["label"].str.cat(df["text"], sep="; ") # Turns input into Topic; topic_description... 
        instruct = f"Represent this {std_str(self.subject)} topic for semantic search: "
        sd_emb = self.model.encode(
            [[instruct, i] for i in input],
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        return sd_emb


class SentenceTransPredictor(TransformerPredictor): 
    def __init__(
        self,
        sd_path: str,
        subject: str,
        model: SentenceTransformer | None = None
    ):  
        self.model = model if model is not None else SentenceTransformer("intfloat/e5-base-v2")
        super().__init__(sd_path, subject)
    
    def embed(self, questions: Sequence[str]):
        """ Uses initialised model to get embeddings on `questions`. """
        qn_emb = self.model.encode(questions, convert_to_tensor=True, normalize_embeddings=True)
        return qn_emb

    def _get_sd_emb(self):
        """ Uses initalised model to get embeddings on initialised study design topics. """
        sd_input = self.sd_df["label"].str.cat(self.sd_df["text"], sep="")
        return self.embed(sd_input)


class TfIdfPredictor:
    def __init__(
        self,
        sd_path: str,
        subject: str,
        vectorizer: TfidfVectorizer | None = None
    ):
        self.vectorizer = vectorizer if vectorizer is not None else TfidfVectorizer()
        self.sd_df = prepare_sd_df(sd_path, subject)
        self.subject = subject

    def run(
        self, 
        exam: str | Path | pd.DataFrame,
        include_report: bool = False,
    ) -> Results:
        """ Runs TF-IDF on an exam and the subject's study design.
        Returns:
            - dataframe containing predictions (optionally with correct labels, if provided)
            - study design topic labels
            - cosine similarity matrix
        """ 
        if isinstance(exam, str) or isinstance(exam, Path):
            qn_df = prepare_exam_df(exam, subject=self.subject, load_report=include_report)
        elif isinstance(exam, pd.DataFrame):
            if include_report:
                raise ValueError("Cannot include report if exam is a pandas DataFrame.")
            qn_df = exam.copy()
        else: 
            raise TypeError("exam must be a str, Path or pandas DataFrame.")

        ans_desc = []
        if not include_report:
            if "comments" in qn_df.columns:          
                qn_df.drop(columns=["comments"])
        else: 
            ans_desc = qn_df["comments"]
        qns_desc = qn_df["text"]
        topic_desc = list(self.sd_df["text"])

        vectorizer = self.vectorizer
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
        pred_df["pred_topic"] = self.sd_df.loc[best_idxs, "label"].reset_index(drop=True)
        pred_df["pred_topic_idx"] = best_idxs

        pred_df["confidence"] = conf_col

        results = Results(pred_df, cos, self.subject, list(self.sd_df["label"]))
        return results


