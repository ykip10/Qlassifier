''' This module contains logic for PDF processings. Some functionalities are:
- Identifying whether or not a PDF has embedded text
- Converting PDF to PNG
- PDF cropping
'''

from typing import List, Tuple, Optional
import os
import json
import re

import pymupdf

FOOTER_PC = 0.15 # Bottom percentage of the pdf which contains the footer. 


def is_scanned(pdf_path: str) -> bool:
    ''' Checks if the pdf is scanned by searching for any embedded text. Returns 
    True if we don't find any embedded text. 
    '''

    doc = pymupdf.open(pdf_path)
    try:
        for page in doc:
            txt = page.get_text("text")
            if txt and txt.strip():
                return False
        return True
    finally:
        doc.close()


def pdf_to_png(pdf_path, output_dir: Optional[str] = None, dpi: int = 150) -> List[str]:
    ''' Convert each page of the PDF to a PNG image and return list of output paths.
    Dpi controls rasterisation resolution of output. 
    '''

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(pdf_path)) or os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    doc = pymupdf.open(pdf_path)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]

    out_paths = []
    
    scale_factor = 72
    zoom = dpi / scale_factor
    mat = pymupdf.Matrix(zoom, zoom)
    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out_name = f"{base_name}_page_{i:03d}.png"
        out_path = os.path.join(output_dir, out_name)
        pix.save(out_path)
        out_paths.append(out_path)
    doc.close()
    return out_paths


def crop_pdf(pdf_path: str, new_coords: Tuple[int] = None, output_path: Optional[str] = None) -> str:
    '''Crop each page of `pdf_path` by specifiying the coordinates of the crop box to be applied. 

    Returns the path to the saved cropped PDF (does not modify the input file
    unless `output_path` points to the same location).

    pdf_path:    Path of pdf to be cropped.
    new_coords:  Coordinates of cropping box to be applied. If empty, does automatic cropping based 
                 on size of the first page of the PDF.
    output_path: Where to save the cropped PDF. If empty, saves in same directory as the orignal.  
    '''
    pdf = pymupdf.open(pdf_path)

    if new_coords is None: 
        page0 = pdf[0]
        w, h = page0.rect.width, page0.rect.height
        new_coords = (w * 0.07, h * 0.04, w * 0.93, (1-FOOTER_PC)*h)  # 

    if output_path is None:
        base = os.path.splitext(os.path.basename(pdf_path))[0]
        output_dir = os.path.dirname(os.path.abspath(pdf_path)) or os.getcwd()
        output_path = os.path.join(output_dir, f"{base}_cropped.pdf")

    x0, y0, x1, y1 = new_coords
    if x0 >= x1 or y0 >= y1: 
        raise ValueError(f"Please input valid cropping coordinates.")

    
    for page in pdf:
        new_rect = pymupdf.Rect(x0, y0, x1, y1)
        # set the crop box so the visible area is reduced
        page.set_cropbox(new_rect)

    pdf.save(output_path)
    pdf.close()
    return output_path


def span_is_bold(span: dict) -> bool:
    ''' Checks if a PyMuPdf span is bold or not. 
    '''
    flags = span.get("flags", 0) or 0
    font = (span.get("font") or "").lower()

    # Check for LaTeX symbols, which might be misclassified as bold. 
    if any(k in font for k in ["cm", "cmsy", "cmex", "stix", "symbol"]):
        return False
    
    return (flags & 16) != 0 or ("bold" in font)


def is_latex(span: dict) -> bool:
    ''' Checks if a PyMuPdf span points to a LaTeX formatted object. 
    '''
    font = (span.get("font") or "").lower()
    return any(k in font for k in ["cm", "cmsy", "cmex", "stix", "symbol"])


def find_final_page(path: str) -> int:
    ''' Find the final examination page by scanning through the footers 
    and looking for an 'end of examination' flag. Input should be uncropped. 
    '''
    try: 
        pdf = pymupdf.open(path)
    except Exception: 
        print(f"Error loading pdf \'{path}\'")
        return None
    
    for page_idx in range(len(pdf)):
        page = pdf[page_idx]
        w, h = page.rect.width, page.rect.height
        new_rect = pymupdf.Rect(0, (1-FOOTER_PC)*h, w, h)
        page.set_cropbox(new_rect)
        footer_props_lst = get_text_and_style(json.loads(page.get_text("json")))
        for text, _, _, in footer_props_lst:
            pattern = r"(?i)^end of (?:\w+\s+)*(questions?|answer|book|examination)\b"
            if re.match(pattern, text):
                print("\n\n HIIIIIIIIII\n\n")
                return page_idx
            
    return page_idx


def get_text_and_style(page_json: dict) -> Tuple[str, bool, bool, int]:
    ''' Given a parsed page.get_text("json") dict, return a list of (text, is_bold, is_latex) tuples.
    '''
    out = []
    for block in page_json.get("blocks", []):
        # text blocks have type == 0
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "")
                bold = span_is_bold(span)
                y0 = span.get("bbox", [0, 0, 0, 0])[1]  # top y-coordinate
                out.append((text, bold, y0))
    return out