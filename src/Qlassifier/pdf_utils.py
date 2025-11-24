from pathlib import Path

import pymupdf
import pandas as pd
from img2table.document import PDF
from img2table.tables.objects.extraction import ExtractedTable

def crop(doc: pymupdf.Document, 
         cr_coords: tuple[int], 
         save_path: str = "") -> pymupdf.Document:
    """ Crop each page of the PDF by specifiying the coordinates of 
    the crop box to be applied.  Returns the croppped PDF document.
    Takes both path and document as 

    doc           : Document to be cropped. 
    scaling_coords: Where w and h are the width and height of the pdf respectively, 
                    we trim the pdf to (w*a1, h*b1, w*a2, h*b2) where 
                    scaling_coords = (a1, b1, a2, b2).
    save_path     : Specifies output save path of cropped image. If empty, doesn't save. 
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
    if save_path:
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

    path            : path of pdf
    pages           : pages of pdf to get tables from. If None, gets all tables.

    """
    pdf = PDF(path, 
              pages=pages,
              detect_rotation=False,
              pdf_text_extraction=True)

    extracted_tables = pdf.extract_tables(implicit_rows=False,
                                          implicit_columns=False,
                                          borderless_tables=False,
                                          min_confidence=50)

                                          
    return extracted_tables


def process_tables(
    tables: dict[int, list[ExtractedTable]]
) -> list[pd.DataFrame]:
    """ Returns list of dataframes: one merged table for all mcq sub-tables, 
    and individual tables for each short answer response. 
    """
    def df_is_mcq(df: pd.DataFrame) -> bool:
        """ Returns True if the dataframe follows the structure of an MCQ report table,
        which typically have more than two rows and can contain empty values in the middle
        columns, where SA tables can not. """
        nrow = df.shape[0]
        middle_cols_df = df[df.columns[1:-1]]
        return nrow > 2 or middle_cols_df.isna().values.any()
    
    mcq_dfs = []
    sa_dfs = []
    for page_tables in tables.values():
        for table in page_tables:
            df = table.df
            if df_is_mcq(df):
                mcq_dfs.append(df)
            else:
                sa_dfs.append(df)

    # We want to merge all the mcq dfs together, table may be split across pages.
    colnames = list(mcq_dfs[0].loc[0]) # First row of first df contains column names 
    mcq_dfs[0] = mcq_dfs[0].drop(index=0)

    if colnames[-1] is None:
        colnames[-1] = "comments"

    for idx, df in enumerate(mcq_dfs):
        df.columns = colnames
        mcq_dfs[idx] = df

    merged = pd.concat(mcq_dfs).reset_index(drop=True)
    # Sometimes, cells spread out in b/w pages. Concatenate comments in this case. 
    none_idx = merged.index[merged[colnames[1]].isna()]
    for idx in none_idx: 
        comment = merged.loc[idx]["comments"]
        merged.loc[idx-1, "comments"] += " " + comment

    merged = merged.dropna(thresh=2).reset_index(drop=True)
    # short answer dfs also need some pre-processing, average column not always 
    # correct 
    for idx, df in enumerate(sa_dfs): 
        new_cols = df.iloc[0].tolist()
        lower = [col.lower() for col in new_cols]
        if any("average" in col for col in lower):
            # theres an average column, always last
            lower[-1] = "average" # Extract float from something like "average/n{float}"
            df.columns = lower 
            curr_s = df.loc[1, "average"].lower().strip()
            df.loc[1, "average"] = curr_s.strip("average").strip()
        df.columns = lower
        df = df[1:].reset_index(drop=True)
        sa_dfs[idx]= df
        
    all_dfs = [merged] + sa_dfs
    return all_dfs