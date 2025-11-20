""" Contains object definitions for our PDF parser, which parses 
a PDF document into a tree. 
"""
from pathlib import Path
import json
import re

import pymupdf

from src.Qlassifier.trees import Tree


class PDFParser:
    """ PDF parsing object. Supports cropping as well as parsing PDF's based on Bold text, 
    which are assumed to be headers, which can the be accessed as a tree object. 
    """
    def __init__(self, path: str, footer_pc: float = 0.15):
        """ Initialise PDF parsing object. 
        path     : Path of pdf to be parsed
        footer_pc: Percentage height of the PDF (from the bottom) to be classified as the "footer"
        """
        self.path = path
        self.footer_pc = footer_pc  # percentage of the bottom of the pdf to be classified as the "footer"
        self._doc = self.load_doc()
        self._copies = []           # keep track of copies to close
        self.qn_hierarchies_regex = [       # question splits, ordered by increasing levels of hierarchy 
                r"^S(?i:ection) [A-Z]", 	# level 1
                r"Question \d+",			# level 2
                r"[a-h]\.",					# level 3
                r"i+\.",					# level 4
        ]

    @property
    def doc(self):
        doc_bytes = self._doc.write()
        new_doc = pymupdf.open(stream=doc_bytes,filetype="pdf")
        self._copies.append(new_doc)
        return new_doc

    def load_doc(self) -> pymupdf.Document:
        try:
            doc = pymupdf.open(self.path)
        except Exception: 
            print(f"Error loading word document \'{self.path}\'")
            doc = None
        return doc

    def split_headings(self) -> Tree:
        """ Extracts text from a PDF using PyMuPdf and sorts them into Sections and Questions.

        Goes through bolded text as candidates for splitting, then checks if the bolded text are 
        valid question splits. Sorts extracted text into a Tree object and returns it. 

        The assumption is that new questions are lablled according to predefined QN_HIERARCHIES. 
        """
        # Find the last page, then crop the pdf. 
        final_page = self._find_final_page()
        doc = self.crop(save=True)

        curr = None
        nodes = []
        page_idx = 0
        while page_idx < len(doc[:(final_page+1)]): 
            page = doc[page_idx]
            txt_props_lst = self._get_text_and_style(json.loads(page.get_text("json")))

            # Extract questions and their text from this page 
            for text, is_bold, y0 in txt_props_lst:
                if is_bold:
                    # possible question split candidate
                    matches = [bool(re.match(pattern, text)) for pattern in self.qn_hierarchies_regex]
                    if any(matches):
                        # Initialise new node
                        curr = Tree(label=text, level=matches.index(True)+1)
                        nodes.append((curr, page_idx, y0))
                elif curr:
                    # ordinary text
                    mark_match = re.search(r"(\d+)\s*marks?", text)
                    if mark_match and not curr.marks:
                        curr.marks = int(mark_match.group(1))
                    curr.text += " " + text
                
            page_idx += 1

        # sort by page then vertical height to get correct ordering  
        nodes.sort(key=lambda t: (t[1], t[2]))  
        nodes = [text for text, page_num, y0 in nodes] 

        root = Tree(label="root", level=0)
        root.build(nodes)
        return root

    def crop(self, new_coords: tuple[int] = None, save: bool = False) -> pymupdf.Document:
        """ Crop each page of the PDF by specifiying the coordinates of the crop box to be applied. 

        Returns the croppped PDF document.

        new_coords:  Coordinates of cropping box to be applied. If empty, does automatic cropping based 
                     on size of the first page of the PDF.
        save:        Boolean flag determining whether or not the cropped output should be saved.  
        """
        new_doc = self.doc

        if new_coords is None:
            page0 = new_doc[0]
            w, h = page0.rect.width, page0.rect.height
            # Let the amount of bottom trimming be scaled by footer_pc
            new_coords = (w * 0.07, h * 0.04, w * 0.93, (1-self.footer_pc)*h)  

        x0, y0, x1, y1 = new_coords
        if x0 >= x1 or y0 >= y1: 
            raise ValueError("Please input valid cropping coordinates.")
        
        # crop each page individually
        for page in new_doc:
            new_rect = pymupdf.Rect(x0, y0, x1, y1)
            page.set_cropbox(new_rect)

        if save:
            path = Path(self.path)
            output_dir = path.resolve().parent
            output_path = output_dir / f"{path.stem}_cropped.pdf"
            new_doc.save(output_path) 

        return new_doc

    def _span_is_bold(self, span: dict) -> bool:
        """ Checks if a PyMuPdf span is bold or not. 
        """
        flags = span.get("flags", 0) or 0
        font = (span.get("font") or "").lower()

        # Check for LaTeX symbols, which might be misclassified as bold. 
        if any(k in font for k in ["cm", "cmsy", "cmex", "stix", "symbol"]):
            return False
        
        return (flags & 16) != 0 or ("bold" in font)

    def _get_text_and_style(self, page_json: dict) -> tuple[str, bool, int]:
        """ Given a parsed page.get_text("json") dict, return a list of (text, is_bold, y0) tuples,
        where y0 is the texts height on the page. 
        """
        out = []
        for block in page_json.get("blocks", []):
            # text blocks have type == 0
            if block.get("type", 0) != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    bold = self._span_is_bold(span)
                    y0 = span.get("bbox", [0, 0, 0, 0])[1]  # top y-coordinate
                    out.append((text, bold, y0))
        return out
    
    def _find_final_page(self) -> int:
        """ Find the index of the final examination page by scanning through the footers 
        and looking for an 'end of examination' flag. Input should be uncropped. 
        """
        for page_idx in range(len(self.doc)):
            # First, crop into only the footer portion of the page
            page = self.doc[page_idx]
            w, h = page.rect.width, page.rect.height
            new_rect = pymupdf.Rect(0, (1-self.footer_pc)*h, w, h)
            page.set_cropbox(new_rect)

            # Search for 'end of examination flag' in footer
            footer_props_lst = self._get_text_and_style(json.loads(page.get_text("json")))
            for text, _, _, in footer_props_lst:
                pattern = r"(?i)^end of (?:\w+\s+)*(questions?|answer|book|examination)\b"
                if re.match(pattern, text):
                    return page_idx
                
        return page_idx

    def is_scanned(self) -> bool:
        """ Checks if the pdf is scanned by searching for any embedded text. Returns 
        True if we don't find any embedded text. 
        """
        for page in self.doc:
            txt = page.get_text("text")
            if txt and txt.strip():
                return False
        return True
    
    def close(self):
        """ Closes PDF and all copies of it.
        """
        if self._doc is not None:
            self._doc.close()
        # Close all copies 
        for doc in self.copies:
            doc.close()
        self._copies.clear()