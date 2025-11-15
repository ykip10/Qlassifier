'''
We scan an input png document, then apply an OCR model for text recognition. 
Optionally, produce an output of the image with the model's predictions.

Usage: python3 src/question_extractor.py path_to_image
Append with -s flag to save an annotated version of the image.
'''

import sys
from pathlib import Path
from typing import List, Tuple
import time

import easyocr
from PIL import Image, ImageDraw, ImageFont


class Section:
    def __init__(self):
        self.questions = [] # Questions belonging to this section sorted by qn number
        self.label = '' # Section label; e.g. "A" for "Section A."


class Question: 
    def __init__(self):
        self.sub_questions = [] # part a), b) etc. 
        self.text = '' 


def annotate_image(image_path: str, results: List[Tuple[List[Tuple[float, float]], str, float]]) -> str:
    '''Draw bounding boxes and text onto the image and save an annotated copy.
    Returns the path to the annotated image.

    image_path: Absolute path to image to be annotated. 
    results:    Results from an OCR output which contains coordinates of text, the corresponding prediction, and it's 
                confidence
    '''

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Try to load a truetype font for nicer labels; fallback to default
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=14)
    except Exception:
        font = ImageFont.load_default()

    for bbox, text, conf in results:
        # bbox formatted as list of four coordinates [(x1, y1), (x2, y2)...]
        xy = bbox
        draw.line(xy + [xy[0]], fill=(255, 0, 0), width=2)
        # place text above top-left corner if possible
        x0, y0 = xy[0]
        label = f"{text} ({conf:.2f})"

        # black box for readability
        tb = draw.textbbox((0, 0), label, font=font)
        text_width = tb[2] - tb[0]
        text_height = tb[3] - tb[1]
        text_size = (text_width, text_height)

        draw.rectangle([x0, y0 - text_size[1] - 4, x0 + text_size[0] + 4, y0], fill=(0, 0, 0))
        draw.text((x0 + 2, y0 - text_size[1] - 2), label, fill=(255, 255, 255), font=font)

    out_path = str(Path(image_path).with_name(Path(image_path).stem + "_ocr.png"))
    img.save(out_path)
    return out_path


def run_ocr(image_path: str, annotate: bool) -> List[Tuple[str]]:
    ''' Runs easyOCR on the image at image_path. 

    image_path: Path of the image to be read. 
    annotate:   Boolean indicator flagging whether or not we should output an annotated image. 
    '''

    # Load and run the model, noting total runtime 
    start_time = time.perf_counter()
    reader = easyocr.Reader(["en"])
    results = reader.readtext(image_path, detail=1)  # list of (bbox, text, conf)
    end_time = time.perf_counter()

    # Extract & print just the text + confidence
    text_lst = []
    for bbox, text, conf in results:
        text_lst.append(text)
        print(f'{text} (conf={conf})')

    # Annotate and save image
    if annotate:
        out = annotate_image(image_path, results)
        print(f"Annotated image written to: {out}")
    print(f"OCR runtime: {end_time-start_time}.")

    return text_lst


def process_ocr(ocr_result: List[str]):
    ''' Processes ocr output to organise text into Section and Question objects. 

    ocr_result: List containing text to be processed. 
    '''
    ocr_result.join()


def main(argv: List[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    if not argv:
        print(__doc__)
        return 2

    annotate = False
    image_path = argv[0]
    try: 
        if argv[1] != "-s":
            print(f"{argv[1]} not a recognised argument.")
            return 1
        annotate = True
    except IndexError:
        pass 
    
    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        return 1

    res = run_ocr(image_path, annotate)
    print(" ".join(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
