""" Module containing helper functions for notebook analyses. 
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def mcq_merged(
    all_reports: dict[str, dict[str, list[pd.DataFrame]]], 
    years: list[int],
) -> dict[str, pd.DataFrame]:
    """ Returns a dictionary mapping each subject to one dataframe consisting
    of all multiple choice tables merged over the years indicated by years.
    """
    all_merged = {}
    subjects = all_reports.keys()
    for subject in subjects:
        report_types = list(all_reports[subject].keys())
        if "math" in subject:
            # only examination 2's have mcq sections
            dfs_to_merge = [all_reports[subject][report][0] for report in report_types \
                            if any(str(year) in report for year in years) and "_2" in report]
        else: 
            dfs_to_merge = [all_reports[subject][report][0] for report in report_types \
                            if any(str(year) in report for year in years)]
        
        merged = pd.concat(dfs_to_merge, axis=0, ignore_index=True)
        all_merged[subject] = merged
    return all_merged


def pie_charts(
    mcqs: dict[str, pd.DataFrame],
    abbrevs: list[str],
    include_r: bool = True,
    guess_correct: bool = True
) -> pd.DataFrame: 
    """ Generates pie charts for 4 subjects' MCQ tables. Also returns the dataframe
    used to construct the data. """
    fig, axs = plt.subplots(2, 2, figsize=(12, 9)) # assuming four subjects

    flat_ax = axs.flatten()
    ax_idx = 0
    subjects = mcqs.keys()
    for subject, abbrev in zip(subjects, abbrevs):
        df = mcqs[subject].copy(deep=True)
        merged = fill_mcq_df(df, include_r, guess_correct)[0]
        
        # Now we plot! 
        explode = [0] * len(merged)
        explode[0] = 0.1
        ax = flat_ax[ax_idx]
        ax.pie(
            merged,
            labels=merged.index,
            autopct="%1.1f%%",
            explode=explode,
            shadow={'ox': -0.04, 'edgecolor': 'none', 'shade': 0.9},
            radius=1.1,
        )
        ax.set_title(f"{abbrev} MCQ Correct Answers")
        ax_idx += 1


def fill_mcq_df(
    df: pd.DataFrame, 
    include_r: bool,
    guess_correct: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ Fills and processes the mcq df's missing "correct_answer" column 
    based on whether or not we include redacted and guess correct 
    answers. Returns the edited dataframe as well as the proportions of 
    each answer. 
    """
    if not guess_correct: 
        # keep only data where we have the correct answer
        df = df.dropna(subset = ["correct_answer"]).reset_index(drop=True)
    else: 
        # let's guess the correct answer based on the mode selection choice
        guessed_answers = get_mode_answers(df)
        df["correct_answer"] = df["correct_answer"].replace("", np.nan)
        df["correct_answer"] = df["correct_answer"].fillna(guessed_answers)

    # Any string that isn't of length 1 should be considered redacted. Label using "R".
    df["correct_answer"] = df["correct_answer"].apply(lambda x: x if len(x) == 1 else "R")

    # let's get the proportions
    proportions = df["correct_answer"].value_counts() / df.shape[0]
    proportions = proportions.sort_index()
    if include_r: 
        return proportions, df\
    
    df = df[df["correct_answer"] != "R"]
    proportions = df["correct_answer"].value_counts() / df.shape[0]
    proportions = proportions.sort_index()
    
    return proportions, df


def get_mode_answers(df: pd.DataFrame) -> pd.Series:
    """ Gets the mode answers for a MCQ df. Returns a series.
    Labels empty strings as 'R' for redacted. """
    option_cols = df.iloc[:, 2:-1].apply(pd.to_numeric, errors="coerce")
    mode_answers = option_cols.apply(
        lambda row: row.idxmax(skipna=True) if row.notna().all() else "R",
        axis = 1
    )
    # now extract upper-case letters from the column labels, convert back to str
    mode_answers = mode_answers.apply(lambda x: x.strip("%").upper())
    return mode_answers