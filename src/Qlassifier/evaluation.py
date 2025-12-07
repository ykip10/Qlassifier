from typing import Sequence

from matplotlib import pyplot as plt
import pandas as pd


def print_sims(qn_idx: int, cos: Sequence):
    """ Prints similarites for a question's predictions. """
    print(f"Question {qn_idx+1} topic similarities")
    sims = list(cos[qn_idx])
    for sim in sorted(sims, reverse=True):
        og_idx = sims.index(sim)
        print(f"Topic {og_idx}: {float(sim):.3f}")


def top3_sims(qn_idx: int, cos: Sequence) -> list[tuple[int, float]]:
    """ Returns a list of tuples containing the top 3 topics with highest similarity
    and their similarity scores. 
    """
    sims = list(cos[qn_idx])
    top3 = [(sims.index(sim), float(sim)) for sim in sorted(sims, reverse=True)[:3]]
    return top3


def correct_in_top3(
    qn_idx: int,
    df: pd.DataFrame,
    cos: Sequence
) -> bool:
    """ Given a DataFrame containing predictions and true topics as well as the
    cosine similarity matrix, checks if the prediction of the topic at qn_idx
    has within the top 3 highest cosine similarity scores.  
    """
    top3_list = top3_sims(qn_idx, cos)
    correct = int(df.loc[ qn_idx, "true_topic_idx"])
    top3_idx = [top3[0] for top3 in top3_list]
    return correct in top3_idx


def evaluate_predictions(df, cos_qn: Sequence, cos_rp: Sequence = None):
    """ Prints evaluation metrics (accuracy and {guess in top3}%). """
    n = len(df)
    print("--EVAL METRICS--")
    ex_correct_pc = len(df[df["true_topic_idx"] == df["pred_topic_idx"]]) / n
    print(f"Exam-only accuracy: {100*ex_correct_pc:.2f}%")

    if cos_rp is not None:
        rp_correct_pc = len(df[df["true_topic_idx"] == df["report_pred"]]) / n
        one_correct_pc = len(
            df[(df["true_topic_idx"] == df["pred_topic_idx"]) |
                (df["true_topic_idx"] == df["report_pred"])]
            ) / n  
        print(f"Report-only accuracy: {100*rp_correct_pc:.2f}%")
        print(f"The percent of time either the report or the exam labels a question correctly is {100*one_correct_pc:.2f}%")
        print("True topic in top 3 report prediction "
             f"{100*(sum([correct_in_top3(qn_idx,df,cos_rp) for qn_idx in range(n)]) / n):.2f}%")

    print("True topic in top 3 question prediction "
        f"{100*(sum([correct_in_top3(qn_idx,df,cos_qn) for qn_idx in range(n)]) / n):.2f}%")


def plot_conf(df):
    """ Creates boxplots of the confidence parameter from a predictions dataframe, grouped by 
    correct/incorrect answers. Used to assess the usefulness of the confidence metric. 
    """
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