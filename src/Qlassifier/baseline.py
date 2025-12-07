import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from src.Qlassifier.utils import prepare_dataframes

def run_tf_idf(
    exam_path: str,
    labels: list[str] = [],
    include_report: bool = True,
    **vectorizerargs
) -> tuple[dict[str, str], np.array]:
    """ Runs TF-IDF on an exam and the subject's study design. Returns 
    dictionary mapping each question in the exam to a study design dot point,
    and the similarity scores for each dot point found in the parsed study design.
    """
    qn_df, sd_df = prepare_dataframes(exam_path)

    ans_desc = []
    if not include_report: 
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
    similarity = cosine_similarity(ex_vecs, topic_vecs)
    best_idxs = np.argmax(similarity, axis=1)

    # add confidence column as similarity / sum(similarity)
    confidence_col = [sims[pred_idx] / sum(sims) for pred_idx, sims in zip(best_idxs, similarity)]
    # Map to [0, 1]
    confidence_col = (confidence_col - min(confidence_col)) / (max(confidence_col) - min(confidence_col))

    pred_df = qn_df.copy()
    pred_df["pred_topic"] = sd_df.loc[best_idxs, "label"].reset_index(drop=True)
    pred_df["pred_topic_idx"] = best_idxs
    if labels: 
        pred_df["true_topic_idx"] = labels 
        pred_df["true_topic"] = sd_df.loc[labels, "label"].reset_index(drop=True)
    pred_df["confidence"] = confidence_col
    return pred_df, similarity


def normalise_question_label(s: str) -> str:
    """ Normalises question labels which become like Question 1.\tc\ti after collapsing.
    Normalises to be of the form Question 1ci. instead. 
    """
    pref = re.search(r"Question \d{1,2}", s).group()
    suff = "".join(c for c in s.replace(pref, "") if c.isalnum())
    return pref + suff + "."