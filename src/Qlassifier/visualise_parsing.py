"""
Module for visualising the parsing of study materials (study designs as Word only, 
examination papers as text-embedded PDF documents only, reports as either PDF/Word), which includes the sorting of
text into hierarchies based on headings and storing the result in trees. For word documents the process works in general, 
but for PDF documents the output only makes sense if it is an examination following VCAA formatting "closely enough."

Usage: 
	python3 -m src.Qlassifier.visualise_parsing path_to_document				# Prints word/PDF doc as a tree 
	python3 -m src.Qlassifier.visualise_parsing path_to_document [target]   	# Searches for 'target' heading, prints target subtree
	python3 -m src.Qlassifier.visualise_parsing path_to_document --show-report  # Processes the document as a report, shows dictionary output.
"""

import sys
from pathlib import Path

from pprint import pprint

from src.Qlassifier.report_processor import ReportProcessor
from src.Qlassifier.word_parser import WordParser
from src.Qlassifier.pdf_parser import PDFParser

def main(argv: list[str] | None = None) -> int:
	argv = argv or sys.argv[1:]
	if not argv or len(argv) > 2:
		print(__doc__)
		return 2

	path = argv[0]
	file_extension = Path(path).suffix

	if file_extension == ".docx":
		parser = WordParser(path)
		root = parser.split_headings()
		root.filter_tree(r"(Q|q)uestion \d.?")
	elif file_extension == ".pdf":
		parser = PDFParser(path, footer_pc=0.15)
		root = parser.split_headings()
	else: 
		print("Unsupported file type. Only parses word documents/PDFs.")
		return 1

	if root is None:
		return 1

	if len(argv) == 2:
		if argv[1] == "--show-report":
			pprint(ReportProcessor(path, ext=file_extension).parse_tables())
			return 0 
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