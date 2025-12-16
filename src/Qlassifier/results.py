from typing import Sequence, Literal

import pandas as pd
import numpy as np 
from torch import Tensor
from matplotlib import pyplot as plt
from sklearn.metrics import (precision_score, recall_score, accuracy_score,
                             f1_score, top_k_accuracy_score)


class Results:
    def __init__(
        self,
        pred_df: pd.DataFrame,
        cos: Sequence[Sequence[float]],
        sd_labels: list[str] | None = None,
        *, 
        correct_topics: list[str] | None = None,
        idx_pred_col: str = "pred_topic_idx"
    ):  
        """ Object containing model results. 

        `pred_df`       : DataFrame containing model predictions.
        `cos`           : Cosine similarity matrix from models predictions.
        `sd_labels`     : Study design topic labels.
        `correct_topics`: Ground truth topic labels, if available.
        `idx_pred_col`  : Which column of `pred_df` contains the topic index prediction.

        Note that many evaluation methods ASSUME the columns of `pred_df`. 
        """
        self._pred_df = pred_df
        # Convert to numpy for compatibility with pandas
        if isinstance(cos, Tensor):
            self.cos = cos.cpu().numpy()
        else:
            self.cos = cos
        self.sd_labels = sd_labels if sd_labels is not None else []
        
        self._correct_topics = correct_topics if correct_topics is not None else []
        self.idx_pred_col = idx_pred_col

        self.qn_labels = pred_df["label"]
    
    @property 
    def pred_df(self):
        return self._pred_df.copy()

    @property
    def correct_topics(self):
        return self._correct_topics
    
    @correct_topics.setter
    def correct_topics(self, value):
        """ Edit `pred_df` to include the new ground truth labels. """
        self._pred_df["true_topic_idx"] = value
        self._pred_df["true_topic"] = self._pred_df["true_topic_idx"].apply(
            lambda x: self.sd_labels[x]
        )
        self._correct_topics = value

    def summary(
        self,
        by: Literal["topic", "overall"],
        subset: Sequence[int] | None = None
    ) -> pd.DataFrame | pd.Series | None: 
        """ Produces a summary of results stored in a DataFrame. 
        
        `by`    : if `by == "topic"`, calculates precision, recall, f1-scores, 
                  counts and proportions for each topic. 
                  if `by == "overall"`, calculates weighted and macro classification metrics, 
                  as well as accuracy and top 3 accuracy. 
        `subset`: Sequence of indices for subsetting the question predictions. 
                  Will only report results for this subset. 
        """
        df = self.pred_df
        if subset is not None:
            df = df.copy()
            df = df.loc[subset]
        df["pred_topic"] = df["pred_topic_idx"].apply(lambda x: self.sd_labels[x])

        # Get metrics in the case where we have labelled data
        if self.correct_topics:
            df["true_topic"] = df["true_topic_idx"].apply(lambda x: self.sd_labels[x])

            topic_counts = df["true_topic"].value_counts().astype(int)
            topic_proportions = topic_counts / sum(topic_counts)
            # Find precision, recall, f1-score and accuracy
            # In the case of overall summary, we use macro-precision 
            # since for single-label multi-class classification, micro-metrics == Accuracy
            scores = []
            scoring_metrics = [precision_score, recall_score, f1_score]
            for metric in scoring_metrics: 
                score = metric(
                    y_true=df["true_topic"],
                    y_pred=df["pred_topic"],
                    labels=self.sd_labels,
                    average=None if by=="topic" else "macro",
                    zero_division=0,
                )
                scores.append(score)
            
        # Need to produce topic-level metrics
        if by == "topic":
            # Create df where columns are sd_labels and rows are metrics
            pred_topic_counts = df["pred_topic"].value_counts().astype(int)
            pred_topic_proportions = pred_topic_counts / sum(pred_topic_counts)
            metrics = pd.DataFrame(
                [pred_topic_counts, pred_topic_proportions],
                index=["Predicted Topic Counts", "Predicted Topic Proportions"],
            ).reindex(self.sd_labels, axis=1, fill_value=0)

            if self.correct_topics:
                score_rows = [pd.Series(score, index=self.sd_labels) for score in scores]
                indices = ["Topic Count", "Topic Proportion", "Precision", 
                           "Recall", "F1-Score"]

                additional_metrics = pd.DataFrame(
                    data=[topic_counts, topic_proportions] + score_rows,
                    index=indices,
                    columns=self.sd_labels               
                )
                metrics = pd.concat([metrics, additional_metrics]).fillna(value=0)

        # Overall metrics (global)
        elif by == "overall":
            if not self.correct_topics:
                print("Cannot summarise with `by==overall` if " \
                      "`Results` initialised with no labels.")
                return None
            accuracy = accuracy_score(
                y_true=df["true_topic"],
                y_pred=df["pred_topic"],
            )
            top3_accuracy = top_k_accuracy_score(
                y_true=df["true_topic_idx"],
                y_score=self.cos,
                k=3,
                normalize=True,
                labels=range(len(self.sd_labels))  
            )

            # Get weighted metrics as well. 
            overall_scores = []
            overall_scoring_metrics = [precision_score, recall_score, f1_score]
            for metric in overall_scoring_metrics: 
                score = metric(
                    y_true=df["true_topic"],
                    y_pred=df["pred_topic"],
                    labels=self.sd_labels,
                    average="weighted",
                    zero_division=0,
                )
                overall_scores.append(score)

            metrics = pd.Series(
                data=[accuracy, top3_accuracy] + scores + overall_scores,
                index=["Accuracy", "Top 3 Accuracy", "Macro-Precision", 
                       "Macro-Recall", "Macro-F1-Score", "Weighted-Precision",
                       "Weighted-Recall", "Weighted-F1-Score"]
            )
        else: 
            raise ValueError("`by` argument can only be one of `['topic', 'overall']`")
    
        return metrics

    def print_all_sims(self, qn_idx: int):
        """ Prints similarites for a question's predictions. """
        sims = list(self.cos[qn_idx])
        for sim in sorted(sims, reverse=True):
            og_idx = sims.index(sim)
            print(f"Topic {og_idx}: {float(sim):.3f}")

    def print_top3s(self):
        """ Prints top 3 results using study design labels. """
        top3s = self.build_top3s(ndigits=3)
        for qn_idx in range(len(top3s)):
            qn = self.qn_labels[qn_idx]
            print(f"{qn}: {top3s[qn_idx]}")
            
    def top3_sims(self, qn_idx: int) -> list[tuple[int, float]]:
        """ Returns a list of tuples containing the top 3 topics with highest similarity
        and their normalised similarity scores. 
        """
        sims = list(self.cos[qn_idx])
        sims_min, sims_max = min(sims), max(sims)
        indexed = list(enumerate(sims))
        indexed_sorted = sorted(indexed, key=lambda x: x[1], reverse=True)[:3]

        top3 = [
            (idx, float((sim - sims_min) / (sims_max - sims_min)))
            for idx, sim in indexed_sorted
        ]
        return top3

    def build_top3s(
        self, 
        ndigits: int | None = None
    ) -> list[list[tuple[str, float]]]:
        """ Finds the top3 predictions of the form `[(topic_prediction, normalised_sim)...] `
        for each question, where the similarity is optionally rounded to 
        `ndigits` decimal points. 
        """
        top3s = []
        for qn_idx in range(len(self.cos)):
            top3 = self.top3_sims(qn_idx)
            # Switch sd-idx for the actual sd label instead,
            if ndigits is not None: 
                top3_with_labels = [(self.sd_labels[sd_idx], round(sim, ndigits)) \
                                     for sd_idx, sim in top3]
            else: 
                top3_with_labels = [(self.sd_labels[sd_idx], sim) \
                                    for sd_idx, sim in top3]
            
            top3s.append(top3_with_labels)
        return top3s

    def correct_in_top3(
        self,
        qn_idx: int,
    ) -> bool:
        """ Given a DataFrame containing predictions and true topics as well as the
        cosine similarity matrix, checks if the prediction of the topic at `qn_idx`
        has within the top 3 highest cosine similarity scores.  
        """
        top3_list = self.top3_sims(qn_idx, self.cos)
        correct = int(self.correct_topics[qn_idx])
        top3_idx = [top3[0] for top3 in top3_list]
        return correct in top3_idx

    def evaluate_predictions(
        self, 
        cos2: Sequence[Sequence[float]] = None,
        second_pred_col: list[str] = "report_pred",
    ):
        """ Prints evaluation metrics (accuracy and {guess in top3}%).
        cos2     : Cosine similarity matrix for prediction column 2 (optional) 
        pred_cols: Column(s) (up to two) which contain topic predictions. 
        """
        pred1 = self.idx_pred_col
        df = self.pred_df
        n = len(df)

        print("--EVAL METRICS--")
        ex_correct_pc = len(df[df["true_topic_idx"] == df[pred1]]) / n
        print(f"{pred1} accuracy: {100*ex_correct_pc:.2f}%")

        if cos2 is not None:
            pred2 = second_pred_col
            # Print data w.r.t second similarity column as well

            # Calculate the accuracy of our second column 
            rp_correct_pc = len(df[df["true_topic_idx"] == df[pred2]]) / n

            # Calculate the amount of times either one of our two columns got the correct topic
            one_correct_pc = len(
                df[
                    (df["true_topic_idx"] == df[pred1]) |
                    (df["true_topic_idx"] == df[pred2])
                ]
            ) / n  
            
            # Print results
            print(f"{pred2} accuracy: {100*rp_correct_pc:.2f}%")
            
            print(
                f"The percent of time either {pred2} or {pred1} "
                f"labels a question correctly is {100*one_correct_pc:.2f}%"
            )
            
            # Additionally print top 3 probability (for the second column)
            print(
                f"% of time the true topic in top 3 {pred2} prediction: " 
                f"{100*(sum([
                    self.correct_in_top3(qn_idx,df,cos2) for qn_idx in range(n)
                ]) / n):.2f}%"
            )

        # Top 3 probability for the first column 
        print(
            f"% of time the true topic in top 3 {pred1} prediction: "
            f"{100*(sum([
                self.correct_in_top3(qn_idx,df,self.cos) for qn_idx in range(n)
            ]) / n):.2f}%"
        )

    def plot_conf(self):
        """ Creates boxplots of the confidence parameter from a predictions dataframe, grouped by 
        correct/incorrect answers. Used to assess the usefulness of the confidence metric. 
        """
        df = self.pred_df
        incorrect_df = df[df["pred_topic_idx"] != df["true_topic_idx"]]
        correct_df = df[df["pred_topic_idx"] == df["true_topic_idx"]]

        _, ax = plt.subplots()

        correct_df["confidence"].plot(kind="box", ax=ax, positions=[1], patch_artist=True,
                                    widths=0.6, boxprops=dict(facecolor="green"))
        incorrect_df["confidence"].plot(kind="box", ax=ax, positions=[2], widths = 0.6, 
                                        patch_artist=True, boxprops=dict(facecolor="red"))

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Correct", "Incorrect"])

        ax.set_title("Confidence of Correct Answers vs. Incorrect Answers")
        ax.set_ylabel("Confidence")

        plt.show()

    def get_qn_labels(self) -> list[str]:
        return self.pred_df["label"]