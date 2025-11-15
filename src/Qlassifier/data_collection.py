'''
Script to collect VCAA past exams and study design for a given VCE subject.
Saves scraped data to ../data/subject_name.

Usage: python3 src/data_collection.py Subject_Name year1,year2,year3...
'''

import sys
import os
import requests
import re
from bs4 import BeautifulSoup
from typing import List
from urllib.parse import urljoin
from itertools import product

VCAA_BASE = "https://www.vcaa.vic.edu.au"
BASE_DIR = os.path.dirname(__file__)

def is_exam(subject, hypertext, years):
    ''' Classify whether or not a hypertext link from a VCAA subject past examinations page 
    (e.g. https://www.vcaa.vic.edu.au/assessment/vce/examination-specifications-past-examinations-and-examination-reports/mathematical-methods)
    contains an examination using VERY simple logic.
    '''
    res = (subject in hypertext) and ("exam" in hypertext or "examination" in hypertext) \
           and ("report" not in hypertext) and ("assessment" not in hypertext) and \
           any([str(year) in hypertext for year in years])
    return res


def extract_exams(subject: str, years: List[int]) -> int: 
    ''' Extracts desired subject's past examinations publicly displayed on the VCAA website. 
    Uses beautiful soup.

    subject: Exact subject name of the VCE/VET subject whose past examinations are to be scraped. 
    years:   Examination years to extract.
    '''

    file_dir = os.path.join(BASE_DIR, "..", f"data/{subject.strip().lower().replace(" ", "_")}")

    url = VCAA_BASE + "/assessment/vce/" \
          "examination-specifications-past-examinations-and-examination-reports/" \
          "examination-specifications-past-examinations-and-external-assessment-reports"
    
    headers = {"User-Agent": "Mozilla/5.0"} # Need header to extract from VCAA 
    
    # Math subjects have two examinations
    is_math = "mathematic" in subject.lower()

    # First, find the subject's exam page
    html = requests.get(url, headers=headers).text
    soup = BeautifulSoup(html, "html.parser")

    found = 0 # Keep track of whether or not we found the subject
    for a in soup.find_all("a", href=True):
        pattern = r"/assessment/(vce|vet)/.+/" + f"{subject.strip().lower().replace(" ", "-")}"
        if re.match(pattern, a["href"]):
            found = 1
            full_url = urljoin(url, a["href"])

    if not found:
        print(f"Unable to find subject {subject}. Make sure it is spelt correctly.")
        return 1
    
    # Now we find any matching examinations. 
    html = requests.get(full_url, headers=headers).text
    soup = BeautifulSoup(html, "html.parser")
    
    found = 0 # Keep track of whether or not we found an exam 
    for a in soup.find_all("a", href=True):
        hypertext = a.get_text(strip=True)
        if is_exam(subject, hypertext, years):
            found = 1
            # Found an exam. Save it. 
            full_url = urljoin(VCAA_BASE, a["href"])
            numbers = re.findall(r"\d+", hypertext)
            year = numbers[0]
            if is_math:
                # Include exam number
                num = numbers[1]
                file_name = f"{year}_{num}.pdf"
            else:
                file_name = f"{year}.pdf"

            os.makedirs(file_dir, exist_ok=True)
            save_path = os.path.join(file_dir, file_name)
            resp = requests.get(full_url, headers=headers)

            with open(save_path, "wb") as f:
                f.write(resp.content)

    if not found:
        print(f"Unable to find an exam for {subject} over the years {years}")
        return 1
    
    return 0 


def extract_sds(subject:str):
    ''' Extracts latest study designs from the VCAA website using Beautiful Soup.

    subject: Subject whose study design is to be extracted.  
    '''
    file_dir = os.path.join(BASE_DIR, "..", f"data/{subject.strip().lower().replace(" ", "_")}")
    os.makedirs(file_dir, exist_ok=True)

    url = "https://www.vcaa.vic.edu.au/curriculum/vce-curriculum/vce-study-designs/vce-study-designs"
    headers = {"User-Agent": "Mozilla/5.0"} # Need header to extract from VCAA 
    
    # Get to subject page
    html = requests.get(url, headers=headers).text
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        pattern = r"/curriculum/.+/" + f"{subject.strip().lower().replace(" ", "-")}" + r"/.+"
        if re.match(pattern, a["href"]):
            full_url = urljoin(url, a["href"])
    print(full_url)
    # Extract study design 
    html = requests.get(full_url, headers=headers).text
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        hypertext = a.get_text(strip=True).lower()
        if "study design" in hypertext:
            full_url = urljoin(VCAA_BASE, a["href"])
            
            save_path = os.path.join(file_dir, f"{subject.strip().lower().replace(" ", "_")}_sd.docx")
            resp = requests.get(full_url, headers=headers)
            with open(save_path, "wb") as f:
                f.write(resp.content)


def main(argv: List[str] | None = None) -> int: 
    argv = argv or sys.argv[1:]
    if not argv or len(argv) > 2:
        print(__doc__)
        return 2
    subject, years = argv
    
    years = years.split(",")

    # Standardise subject_name to Subject Name.
    subject = subject.replace("_", " ")
    subject = subject.replace(subject[0], subject[0].upper(), 1)
    if " " in subject:
        idx = subject.index(" ") + 1
        subject = subject[:idx] + subject[idx].upper() + subject[idx+1:]

    if extract_exams(subject, years): 
        return 1
    extract_sds(subject)
    return 0 


if __name__ == "__main__": 
    raise SystemExit(main())