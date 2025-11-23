from pathlib import Path

import pymupdf
import pandas as pd
from img2table.document import PDF
from img2table import ExtractedTable


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
    mark distribution tables at the given pages as a list of pandas dataframes.
    Performs some post-processing.

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


def preprocess_table(
    tables: dict[int, list[ExtractedTable]],
    pages: list[int]
) -> pd.DataFrame:
    """ Returns merged mcq tables and the page index where short answer questions begin. """
    mcq_dfs = []

    def df_is_mcq(df: pd.DataFrame) -> bool:
        """ Returns True if the dataframe follows the structure of an MCQ report table,
        which typically have more than two rows and can contain empty values in the middle
        columns, where SA tables can not. """
        nrow = df.shape[0]
        middle_cols_df = df[df.columns[-1:1]]
        return nrow > 2 or any(middle_cols_df.isna())
    

    # assuming mcq comes first 
    for page_num in pages:
        page_tables = tables[page_num]
        for table in page_tables:
            df = table.df
            if df_is_mcq(df):
                mcq_dfs.append(df)

    colnames = list(mcq_dfs[0].loc[0]) # First row of first df contains column names 
    mcq_dfs[0] = mcq_dfs[0].drop(index=0)

    if colnames[-1] is None:
        colnames[-1] = "comments"

    for idx, df in enumerate(mcq_dfs):
        df.columns = colnames
        mcq_dfs[idx] = df

    merged = pd.concat(mcq_dfs).reset_index(drop=True)
    none_idx = merged.index[merged[colnames[0]].isna()]
    for idx in none_idx: 
        comment = merged.loc[idx]["comments"]
        merged.loc[idx-1, "comments"] += " " + comment

    merged = merged.dropna(thresh=2).reset_index(drop=True)
    # Now, concatenate commetns column of rows with 
    return merged