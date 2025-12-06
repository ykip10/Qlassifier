import pandas as pd
import numpy as np

def print_sims(qn_idx: int, cos: np.array):
    """ Prints similarites for a question's predictions. """
    print(f"Question {qn_idx+1} topic similarities")
    sims = list(cos[qn_idx])
    for sim in sorted(sims, reverse=True):
        og_idx = sims.index(sim)
        print(f"Topic {og_idx}: {float(sim):.3f}")


def top3_sims(qn_idx: int, cos: np.array) -> list[tuple[int, float]]:
    """ Returns a list of tuples containing the top 3 topics with highest similarity
    and their similarity scores. 
    """
    sims = list(cos[qn_idx])
    top3 = [(sims.index(sim), float(sim)) for sim in sorted(sims, reverse=True)[:3]]
    return top3


def correct_in_top3(
    qn_idx: int,
    df: pd.DataFrame,
    cos: np.array
) -> bool:
    """ Given a DataFrame containing predictions and true topics as well as the
    cosine similarity matrix, checks if the prediction of the topic at qn_idx
    has within the top 3 highest cosine similarity scores.  
    """
    top3_list = top3_sims(qn_idx, cos)
    correct = int(df.loc[ qn_idx, "true_topic_idx"])
    top3_idx = [top3[0] for top3 in top3_list]
    return correct in top3_idx