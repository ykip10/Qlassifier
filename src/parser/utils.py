""" Parsing utilities. """
from pathlib import Path
import re

import pymupdf
import pandas as pd
import numpy as np
from img2table.document import PDF
from img2table.tables.objects.extraction import ExtractedTable

def crop(
    doc: pymupdf.Document,     
    cr_coords: tuple[int], 
    save_path: str | None = None
) -> pymupdf.Document:
    """ Crop each page of the PDF by specifiying the coordinates of 
    the crop box to be applied. Returns the croppped PDF document.

    `doc`           : Document to be cropped. 
    `scaling_coords`: Where w and h are the width and height of the pdf respectively, 
                      we trim the pdf to `(w*a1, h*b1, w*a2, h*b2)` where 
                      scaling_coords = `(a1, b1, a2, b2)`.
    `save_path`     : Specifies output save path of cropped image. If empty, doesn't save. 
    """
    if cr_coords is None:
        return doc
    
    page0 = doc[0]
    w, h = page0.rect.width, page0.rect.height
    # Let the amount of bottom trimming be scaled by footer_pc
    new_coords = [i*j for i, j in zip([w, h, w, h], cr_coords)]
    x0, y0, x1, y1 = new_coords
    if x0 >= x1 or y0 >= y1: 
        raise ValueError("Please input valid cropping coordinates.")
    
    # crop each page individually
    for page in doc:
        new_rect = pymupdf.Rect(x0, y0, x1, y1)
        page.set_cropbox(new_rect)
    if save_path is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        doc.save(save_path)
    
    return doc


def get_tables(
    path: str,
    pages: list[int] = None
) -> dict[int, list[ExtractedTable]]:
    """ Extracts tables from the PDF at path which are assumed to be 
    mark distribution tables at the given pages. returns a dictionary
    mapping page indices to a list of the tables found on that page. 

    `path` : path of pdf
    `pages`: pages of pdf to get tables from. If None, gets all tables.
    """
    pdf = PDF(
        path, 
        pages=pages,
        detect_rotation=False,
        pdf_text_extraction=True
    )
    extracted_tables = pdf.extract_tables(
        implicit_rows=False,
        implicit_columns=False,
        borderless_tables=False,
        min_confidence=50
    )
    return extracted_tables


def convert_extracted_tables(
    tables: dict[int, list[ExtractedTable]],
) -> tuple[
    list[pd.DataFrame],
    list[pd.DataFrame]
]:
    """ Converts img2table PDF table extraction output into two lists of dataframes,
    one merged table for mcq answers, and one for each non-mcq. Outputs are without 
    an assumed heading.
    """
    def df_is_mcq(df: pd.DataFrame) -> bool:
        """ Returns `True` if the dataframe follows the structure of an MCQ report table,
        which typically have more than two rows and can contain empty values in the middle
        columns, where SA tables can not.
        """

        # "%" in columns only occurs for mcq tables
        columns = df.loc[0, df.columns[:-1]]
        str_cols = all(isinstance(col, str) for col in columns)
        has_percentage_cols = any("%" in str(col) for col in columns) \
                              if str_cols else False
        
        has_marks_col = any(re.match("(M|m)arks?", str(col)) for col in columns) \
                        if str_cols else False
        
        # All columns of a row except the last contain digits if and only if 
        # the row belongs to an mcq table, by standard VCAA formatting. 
        all_digit_cols = all(str(col).strip().isdigit() for col in columns
                             if col is not None)
        return (has_percentage_cols or all_digit_cols) and not has_marks_col
    
    mcq_dfs = []
    sa_dfs = []
    for page_tables in tables.values():
        for table in page_tables:
            df = table.df
            if len(df.columns) < 4:
                # not a relevant table
                continue
            if df_is_mcq(df):
                mcq_dfs.append(df)
            else:
                sa_dfs.append(df)

    return mcq_dfs, sa_dfs
    

def process_tables(
    mcq_dfs: list[pd.DataFrame],
    sa_dfs: list[pd.DataFrame],
) -> list[pd.DataFrame]:
    """ Returns list of dataframes: one merged table for all mcq sub-tables, 
    and individual tables for each short answer response. 

    `mcq_dfs`    : List of dataframes representing tables which can be assumed to come
                   from multiple choice report answers.
    `sa_dfs`     : List of dataframes representing non-mcq tables. 
    """
    # We want to merge all the mcq dfs together, table may be split across pages.
    if mcq_dfs:
        colnames = list(mcq_dfs[0].loc[0])
        mcq_dfs[0] = mcq_dfs[0].drop(index=0)

        # assume empty last column contains comments. 
        # This is a common vcaa formatting choice. 
        if colnames[-1] is None:
            colnames[-1] = "comments"

        # For the options columns, we dont want any spaces
        new_cols = [col.lower().strip().replace(" ", "_") if "%" not in col \
                    else col.lower().strip().replace(" ", "") for col in colnames]
        
        # First column gives the question number. Want to standardise this
		# to always be "question" as opposed to "Qn", "qn." etc.
        new_cols[0] = "question"

        for idx, df in enumerate(mcq_dfs):
            # drop rows that exactly match the inital header row
            header = pd.Series(colnames, index=df.columns)
            df = df.loc[~(df == header).all(axis=1)]
            
            # add in correct answer column
            correct_ans_col = pd.DataFrame(
                {"correct_answer": [np.nan]*df.shape[0]}
            )
            df = pd.concat([df, correct_ans_col], axis=1, ignore_index=True)
            df.columns = new_cols + ["correct_answer"]
            mcq_dfs[idx] = df

        merged = pd.concat(mcq_dfs).dropna(how="all").reset_index(drop=True)
        # Sometimes, cells spread out in b/w pages. Concatenate comments in this case. 
        none_idx = merged.index[merged[new_cols[0]].isna()]
        for idx in none_idx: 
            comment = merged.loc[idx, "comments"]
            if comment is None or pd.isna(comment):
                # Empty comments; nothing to concatenate. 
                continue
            merged.loc[idx-1, "comments"] += " " + str(comment)

        # Drop rows with empty missing values in questions column. 
        merged = merged.dropna(subset=new_cols[0]).reset_index(drop=True)
        merged = standardise_mcq_df(merged)
    else:
        merged = None
    
    # short answer dfs also need some pre-processing, average column not always 
    # correct. Also, some tables which aren't actually SA tables may have been
    # picked up. 
    new_sa_dfs = []
    for idx, df in enumerate(sa_dfs): 
        new_cols = df.iloc[0].tolist()
        if not any(bool(re.match("(M|m)ark", string=col)) for col in new_cols if col is not None):
            # not a marks distribution table if it doesn't have a "Marks" col
            continue

        lower = [col.lower().strip() for col in new_cols]
        if any("average" in col for col in lower):
            # theres an average column, always last
            lower[-1] = "average" # Extract float from something like "average\n{float}"
            df.columns = lower
            curr_s = df.loc[1, "average"].lower().strip()
            df.loc[1, "average"] = curr_s.strip("average").strip()

        # set new heading, remove first row 
        df.columns = lower
        df = df[1:].reset_index(drop=True)
        new_sa_dfs.append(df)
    
    sa_dfs = new_sa_dfs
    all_dfs = [merged] + sa_dfs if merged is not None else sa_dfs
    return all_dfs


def standardise_mcq_df(
    df: pd.DataFrame,
    edit_correct_ans_col: bool = True
) -> pd.DataFrame:
    """ For consistency between mcq dfs we drop n/a column, and ensure that the 
    correct answer column always appears after the question number (this is checked if 
    `edit_correct_ans_col == True`).
    """
    cols = df.columns
    if edit_correct_ans_col:
        # reorder columns
        option_idx = ["%" in col for col in cols].index(True)
        reordered_cols = list(cols[:option_idx]) + [cols[-1]] + list(cols[option_idx:-1])
        df = df.reindex(columns=reordered_cols)
        cols = df.columns
    
    # Drop na column
    na_matches = [bool(re.match("%( |_)?(N|n).+", col)) for col in cols]
    if any(na_matches):
        match_idx = na_matches.index(True)
        match_col = df.columns[match_idx]
        df = df.drop(columns=[match_col])

    return df 