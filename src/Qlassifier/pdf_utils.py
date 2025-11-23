from pathlib import Path


import pymupdf
import pandas as pd
from gmft_pymupdf import PyMuPDFDocument
from gmft.auto import AutoTableFormatter, AutoTableDetector

import time
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
) -> list[pd.DataFrame]:
    """ Extracts tables from the PDF at path which are assumed to be 
    mark distribution tables at the given pages as a list of pandas dataframes.
    Performs some post-processing.

    path            : path of pdf
    pages           : pages of pdf to get tables from. If None, gets all tables.

    """
    detector = AutoTableDetector()
    formatter = AutoTableFormatter()

    tables = [] # list[CroppedTable], see gmft 
    page_idxs = []

    doc = PyMuPDFDocument(path)
    if pages is None: 
        pages = range(len(doc))
    
    for page_idx in pages:
        page = doc[page_idx]
        extracted = detector.extract(page)
        tables += extracted
        page_idxs.append((page_idx, len(extracted)))

    dfs = [formatter.extract(table).df() for table in tables]
    doc.close()

    # Perform some post-processing
    def fix_index(df: pd.DataFrame, index: int) -> pd.DataFrame:
        """ The detection and formatting process often collapses 
        two adjacent rows into one. This function fixes these 
        concatenation issues by splitting the concatenated row into two. 
        """
        concatd_vals = df.iloc[index].values[:-1] # Exclude comments column

        # Separate concatenated column into two 
        sepd_strs = [unseparated_str.split() for unseparated_str in concatd_vals] # separated strings
        str1_lst = []
        str2_lst = []
        for sepd_str in sepd_strs:
            str1_lst.append(sepd_str[0])
            str2_lst.append(sepd_str[1])

        # Add in value for comments column. It is probable that 
        # the concatenation would not have
        # occurred given the comments column were occupied,
        # but it is nonetheless possible we are losing information here. 
        str1_lst += [None]
        str2_lst += [None]

        # insert split strings into correct position
        df.loc[index] = str1_lst
        df.loc[index+0.5] = str2_lst

        # Finally re-sort index
        df = df.sort_index().reset_index().drop(columns=["index"])
        return df
          
    for i, df in enumerate(dfs):        
        # Index 1 is guaranteed to have a "mark" value 
        # (since indices 0 and -1 are the only non mark columns)
        # which should just be one number, if no concatenation 
        # issues occurred. 
        mark_col = df.columns[1] 

        # Remove empty rows
        df.dropna(axis=0, thresh=3, inplace=True) 

        # Contains multiple words => unintended concatenation
        problem_indices = df[mark_col].iloc[
            list(df[mark_col].apply(lambda x: " " in x.strip())) 
        ].index 

        for idx in problem_indices:
            df = fix_index(df, int(idx))
            dfs[i] = df 
    return dfs

