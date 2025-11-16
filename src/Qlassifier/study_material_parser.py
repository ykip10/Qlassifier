''' 
Tools for processing study designs (as word documents) and examination papers (as text-embedded PDF documents)
and sorting text into hierarchies based on headings. For word documents the process works in general, 
but for PDF documents the output only makes sense if it is a vcaa past examination. Supports exact search.

Usage: 
	python3 study_material_parser.py path_to_document			 # Prints word doc as a tree 
	python3 study_material_parser.py path_to_document [target]   # Searches for 'target' heading, prints subtree
'''

import sys
import os
import re
import json
from typing import List, Tuple

import pdf_utils
import trees
import pymupdf
from docx import Document

QN_HIERARCHIES_REGEX = [    # question splits, ordered by increasing levels of hierarchy 
    r"SECTION \d", 		# level 1
    r"Question \d+",	# level 2
    r"[a-z]\.",			# level 3
    r"i+.",				# level 4
]

def extract_headings(path: str) -> List[Tuple[int, str]]:
	''' Return a list of (level, text) for headings found in the document. 
	It operates on the assumption that the document's headings are stylised as headings
	(as opposed to something like boldness/font size).
	'''
	try:
		doc = Document(path)
	except Exception: 
		print(f"Error loading word document \'{path}\'")
		return None
	
	headings = []
	# First, build the headings
	curr = None
	for p in doc.paragraphs:
		style_name = getattr(p.style, "name", "") or ""
		text = p.text.strip()
		if not text:
			continue
		if "heading" in style_name.lower():
			# In word formatting, typically we have styles of the form "Heading 2"
			# The "Heading" word implies its a heading font. The number displays 
			# the hierarchy.
			parts = style_name.split()
			for part in parts[::-1]:
				# Extract number
				if part.isdigit():
					level = int(part)
					break
			curr = trees.Tree(label=text, level=level)
			headings.append(curr)
		else: 
			if curr: 
				curr.text += " " + text

	# Build the tree then retunr
	root = trees.Tree(label="root", level=0)
	root.build(headings)
	return root


def process_pdf(path):
	''' Extracts text from pymupdf Document object and sorts them into Sections and Questions.
	Goes through bolded text as candidates for splitting, then checks if the bolded text are 
	valid question splits. Sorts extracted text into a Tree object and returns it. 
	'''
	try: 
		pdf = pymupdf.open(path)
	except Exception: 
		print(f"Error loading pdf \'{path}\'")
		return None

	curr = None
	nodes = []
	for page in pdf[1:]: # Skip first page (cover page)
		txt_bold_lst = pdf_utils.extract_text_and_bold(json.loads(page.get_text("json")))
		for text, is_bold in txt_bold_lst:
			if is_bold:
				# question split candidate
				matches = [bool(re.match(pattern, text)) for pattern in QN_HIERARCHIES_REGEX]
				if any(matches):
						# Found question split
						if curr: 
							nodes.append(curr)
						curr = trees.Tree(label=text, level=matches.index(True)+1)
						continue
			if curr:
				curr.text += " " + text

	root = trees.Tree(label="root", level=0)
	root.build(nodes)
	return root


def main(argv: List[str] | None = None) -> int:
	argv = argv or sys.argv[1:]
	if not argv or len(argv) > 2:
		print(__doc__)
		return 2

	path = argv[0]
	_, file_extension = os.path.splitext(path)
	
	if file_extension == ".docx":
		root = extract_headings(path)
	elif file_extension == ".pdf":
		root = process_pdf(path)
	else: 
		print("Unsupported file type. Only parses word documents/PDFs.")
		return 1
	
	if root is None:
		return 1

	if len(argv) == 2:
		# Need to execute search 
		target = argv[1]
		root = root.label_search(target)
		if not root:
			print(f"Unable to find {target} in tree.")
			return 1
	root.print_tree()
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
