''' This module contains logic for PDF processings. Some functionalities are:
- Identifying whether or not a PDF has embedded text
- Converting PDF to PNG
- PDF cropping
'''

from typing import List, Tuple, Optional
import os

import pymupdf

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


def crop_pdf(pdf_path: str, new_coords: Tuple[int], output_path: Optional[str] = None) -> str:
    '''Crop each page of `pdf_path` by removing the given amounts from the
    left, top, right and bottom edges. Values are assumed to be in PDF
    points (pt) — the standard unit used by PDFs (1 pt = 1/72 inch).

    Returns the path to the saved cropped PDF (does not modify the input file
    unless `output_path` points to the same location).
    '''
    if output_path is None:
        base = os.path.splitext(os.path.basename(pdf_path))[0]
        output_dir = os.path.dirname(os.path.abspath(pdf_path)) or os.getcwd()
        output_path = os.path.join(output_dir, f"{base}_cropped.pdf")

    x0, y0, x1, y1 = new_coords
    if x0 >= x1 or y0 >= y1: 
        raise ValueError(f"Please input valid cropping coordinates.")

    doc = pymupdf.open(pdf_path)
    for page in doc:
        new_rect = pymupdf.Rect(x0, y0, x1, y1)
        # set the crop box so the visible area is reduced
        page.set_cropbox(new_rect)

    doc.save(output_path)
    doc.close()
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


def extract_text_and_bold(page_json: dict):
    ''' Given a parsed page.get_text("json") dict, return a list of (text, is_bold) tuples.
    '''
    out = []
    for block in page_json.get("blocks", []):
        # text blocks have type == 0
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                out.append((span.get("text", ""), span_is_bold(span)))
    return out