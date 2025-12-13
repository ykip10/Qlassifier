from typing import Sequence

from matplotlib import pyplot as plt
import pandas as pd


def print_sims(qn_idx: int, cos: Sequence[Sequence[float]]):
    """ Prints similarites for a question's predictions. """
    print(f"Question {qn_idx+1} topic similarities")
    sims = list(cos[qn_idx])
    for sim in sorted(sims, reverse=True):
        og_idx = sims.index(sim)
        print(f"Topic {og_idx}: {float(sim):.3f}")


def top3_sims(qn_idx: int, cos: Sequence[Sequence[float]]) -> list[tuple[int, float]]:
    """ Returns a list of tuples containing the top 3 topics with highest similarity
    and their normalised similarity scores. 
    """
    sims = list(cos[qn_idx])
    sims_min = min(sims)
    sims_max = max(sims)
    indexed = list(enumerate(sims))
    indexed_sorted = sorted(indexed, key=lambda x: x[1], reverse=True)[:3]

    top3 = [
        (idx, float((sim - sims_min) / (sims_max - sims_min)) if sims_max != sims_min else 1.0)
        for idx, sim in indexed_sorted
    ]
    return top3


def build_top3s(
    sd_labels: pd.Series,
    cos: Sequence[Sequence[float]],
    ndigits: int | None = None
) -> list[list[tuple[str, float]]]:
    """ Finds the top3 predictions of the form [(topic_prediction, normalised_sim)...] 
    for each question, where the similarity is optionally rounded to `ndigits` decimal points. """
    top3s = []
    for qn_idx in range(len(cos)):
        top3 = top3_sims(qn_idx, cos)
        # Switch sd-idx for the actual sd label instead,
        if round is not None: 
            top3_with_labels = [(sd_labels[sd_idx], round(sim, ndigits)) for sd_idx, sim in top3]
        else: 
            top3_with_labels = [(sd_labels[sd_idx], sim) for sd_idx, sim in top3]
        top3s.append(top3_with_labels)
    return top3s


def correct_in_top3(
    qn_idx: int,
    df: pd.DataFrame,
    cos: Sequence[Sequence[float]]
) -> bool:
    """ Given a DataFrame containing predictions and true topics as well as the
    cosine similarity matrix, checks if the prediction of the topic at qn_idx
    has within the top 3 highest cosine similarity scores.  
    """
    top3_list = top3_sims(qn_idx, cos)
    correct = int(df.loc[qn_idx, "true_topic_idx"])
    top3_idx = [top3[0] for top3 in top3_list]
    return correct in top3_idx


def evaluate_predictions(
    df: pd.DataFrame,
    cos1: Sequence[Sequence[float]],
    cos2: Sequence[Sequence[float]] = None,
    pred_cols: list[str] = ["pred_topic_idx", "report_pred"],
):
    """ Prints evaluation metrics (accuracy and {guess in top3}%).
    df       : Prediction dataframe containing topic predicitons and true labels. 
    cos1     : Cosine similarity matrix for prediction column 1
    cos2     : Cosine similarity matrix for prediction column 2 (optional) 
    pred_cols: Column(s) (up to two) which contain topic predictions. 
    """
    if not pred_cols or len(pred_cols) > 2: 
        raise ValueError("pred_cols argument must be a list of length 1 or 2.")

    pred1 = pred_cols[0]
    if len(pred_cols) == 2:
        pred2 = pred_cols[1]

    n = len(df)
    print("--EVAL METRICS--")
    ex_correct_pc = len(df[df["true_topic_idx"] == df[pred1]]) / n
    print(f"{pred1} accuracy: {100*ex_correct_pc:.2f}%")

    if cos2 is not None:
        rp_correct_pc = len(df[df["true_topic_idx"] == df[pred2]]) / n
        one_correct_pc = len(
            df[(df["true_topic_idx"] == df[pred1]) |
                (df["true_topic_idx"] == df[pred2])]
            ) / n  
        
        print(f"{pred2} accuracy: {100*rp_correct_pc:.2f}%")
        
        print(f"The percent of time either {pred2} or {pred1}"
              f"labels a question correctly is {100*one_correct_pc:.2f}%")
        
        print(f"% of time the true topic in top 3 {pred2} prediction: " 
              f"{100*(sum([correct_in_top3(qn_idx,df,cos2) for qn_idx in range(n)]) / n):.2f}%")

    print(f"% of time the true topic in top 3 {pred1} prediction: "
        f"{100*(sum([correct_in_top3(qn_idx,df,cos1) for qn_idx in range(n)]) / n):.2f}%")


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