"""
Script to collect VCAA past exams and study design for a given VCE subject.
Saves scraped data to ../data/subject_name.

Usage: python3 -m src.Qlassifier.material_collector subject_name year1,year2,year3...
"""

import sys
from pathlib import Path
import re
from typing import Sequence
from urllib.parse import urljoin, urlparse
import time 

import requests
from bs4 import BeautifulSoup

from src.Qlassifier.paths import DATA_DIR

VCAA_BASE = "https://www.vcaa.vic.edu.au"


def vcaa_extract_exam_materials(
    subject: str,
    years: Sequence[int],
    reports=True,
    exams=True
) -> int: 
    """ Extracts desired subject's past examinations publicly displayed on the VCAA website. 
    Uses beautiful soup.

    subject: Exact subject name of the VCE/VET subject whose past examinations are to be scraped. 
    years  : Examination years to extract.
    reports: Whether or not to extract assessment reports. 
    exams  : Whether or not to extract examination documents. 
    """
    is_math = "mathematic" in subject.lower() # Math subjects have two examinations
    headers = {"User-Agent": "Mozilla/5.0"}   # Need header to extract from VCAA 
    file_dir = DATA_DIR / subject.strip().lower().replace(" ", "_")

    full_url = find_exams_page(subject, headers=headers)
    # Now we find any matching examinations. 
    html = requests.get(full_url, headers=headers).text
    soup = BeautifulSoup(html, "html.parser")

    # We extract hyperlinks which match VCAA hypertext naming convention
    year_pattern = "|".join(map(str, years))

    if exams:
        exam_pattern = re.compile(
            rf"^(?=.*\bexam(in(a|i)nation)?\b)(?!.*\breport\b)(?!.*\bassessment\b)(?=.*\b({year_pattern})\b)",
            re.IGNORECASE,
        )
        exam_dir = file_dir / "past_exams"
        
        if not save_link(soup, exam_pattern, exam_dir, is_math):
            print(f"Unable to find an exam for {subject} over the years {years}")
            return 0

    if reports: 
        # Need to save reports
        report_pattern = re.compile(
            rf"^(?=.*\bexam(in(a|i)nation)?\b)(?=.*\breport\b)(?=.*\b({year_pattern})\b)",
            re.IGNORECASE,
        )
        report_dir = file_dir / "past_reports"

        if not save_link(soup, report_pattern, report_dir, is_math):
            print(f"Problem finding reports exam for {subject} over the years {years}")
            return 0
    return 1


def extract_sds(subject: str) -> int:
    """ Extracts latest study designs from the VCAA website using Beautiful Soup.

    subject: Subject whose study design is to be extracted.  
    """
    subject = subject.replace("_", " ")

    file_dir = DATA_DIR / subject.strip().lower().replace(" ", "_") / "study_design"
                            
    file_dir.mkdir(parents=True, exist_ok=True)

    url = "https://www.vcaa.vic.edu.au/curriculum/vce-curriculum/vce-study-designs/vce-study-designs"
    headers = {"User-Agent": "Mozilla/5.0"} # Need header to extract from VCAA 
    
    # Get to subject curriculum page
    html = requests.get(url, headers=headers).text
    soup = BeautifulSoup(html, "html.parser")
    subject_pattern = re.compile(rf"/curriculum/.+/{subject.strip().lower().replace(" ", "-")}/.+")

    try:
        full_url = urljoin(url, soup.find_all("a", href=subject_pattern)[0]["href"])
    except IndexError: 
        print(f"Unable to find subject {subject} study design. Make sure it is spelt correctly")
        return 0
    
    # Extract study design 
    html = requests.get(full_url, headers=headers).text
    soup = BeautifulSoup(html, "html.parser")

    sd_name = "mathematics" if "mathematic" in subject else subject # math subjects sd have diff naming convention
    pattern = re.compile(rf"{sd_name} study design", re.IGNORECASE)

    matches = soup.find_all("a", string=pattern, href=True)
    if not matches: 
        print(f"Couldn't find any study designs for {subject}. Are you sure it has a study design?")

    for a in matches:
        full_url = urljoin(VCAA_BASE, a["href"])
        save_path = file_dir / f"{subject.strip().lower().replace(" ", "_")}_sd.docx"
        if save_path.exists(): # don't re-download if already exists
            return 1
        
        resp = requests.get(full_url, headers=headers)
        with open(save_path, "wb") as f:
            f.write(resp.content)

    return 1


def save_link(
    soup,
    pattern: str,
    save_dir: str,
    is_math: bool = 0,
    headers: str = {"User-Agent": "Mozilla/5.0"}
) -> int:
    """ Saves an exam/report/study-design pointed to by full_url with hypertext text in predetermined data folder.
    
    soup    : Beautiful soup object for the page to be scraped 
    pattern : Pattern of hypertext we want to scrape from
    save_dir: Directory the file should be saved in. 
    is_math : Is the subject being scraped a math one.
    headers : Headers to use while scraping. 

    Returns 0 if no links were found, 1 otherwise. 
    """
    links = soup.find_all("a", string=pattern, href=True)
    if not links:
        return 0
    # Save any links we've scraped 
    for a in links: 
        text = a.get_text(strip=True)
        full_url = urljoin(VCAA_BASE, a["href"])
        numbers = re.findall(r"\d+", text)
        
        year = numbers[0]
        num = numbers[1] if is_math else None
        
        # Get file extension
        parsed_url = urlparse(full_url)
        ext = Path(parsed_url.path).suffix

        # set up directory
        file_name = f"{year}_{num}{ext}" if is_math else f"{year}{ext}"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / file_name

        if save_path.exists(): # already saved
            continue
        start_time = time.perf_counter()
        # download and save 
        resp = requests.get(full_url, headers=headers)
        with open(save_path, "wb") as f:
            f.write(resp.content)
    return 1


def find_exams_page(subject: str, headers: dict[str, str]) -> str:
    """ Finds the URL of the exams page for the given subjects."""
    
    # Standardise subject input to "Subject Name".
    subject = subject.lower().replace("_", " ")
    subject = subject.replace(subject[0], subject[0].upper(), 1)   
    if " " in subject:
        idx = subject.index(" ") + 1
        subject = subject[:idx] + subject[idx].upper() + subject[idx+1:] 
    
    # could become function argument if the need arises
    url = VCAA_BASE + "/assessment/vce/" \
          "examination-specifications-past-examinations-and-examination-reports/" \
          "examination-specifications-past-examinations-and-external-assessment-reports" 
    
    # ===== START SCRAPING ===== #
    # First, find the subject's exam page
    html = requests.get(url, headers=headers).text
    soup = BeautifulSoup(html, "html.parser")
    subject_pattern = re.compile(rf"^/assessment/(vce|vet)/.+/{subject.strip().lower().replace(" ", "-")}")
    
    # Try to find the subject's page
    try:
        full_url = urljoin(url, soup.find_all("a", href=subject_pattern)[0]["href"])
    except IndexError as IE: 
        print(f"Unable to find subject {subject}. Make sure it is spelt correctly.")
        return IE

    return full_url


def main(argv: list[str] | None = None) -> int: 
    argv = argv or sys.argv[1:]
    if not argv or len(argv) > 2:
        print(__doc__)
        return 2
    
    subject, years = argv
    years = years.split(",")

    if not vcaa_extract_exam_materials(subject, years): 
        return 1
    if not extract_sds(subject):
        return 1

    return 0 


if __name__ == "__main__": 
    raise SystemExit(main())