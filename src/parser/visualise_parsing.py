"""
Module for visualising the parsing of study materials (study designs as Word only, 
examination papers as text-embedded PDF documents only, reports as either PDF/Word), which includes the sorting of
text into hierarchies based on headings and storing the result in trees. For word documents the process works in general, 
but for PDF documents the output only makes sense if it is an examination following VCAA formatting "closely enough."

Usage: 
	python3 -m src.parser.visualise_parsing path_to_document				# Prints word/PDF doc as a tree 
	python3 -m src.parser.visualise_parsing path_to_document [target]   	# Searches for 'target' heading, prints target subtree
	python3 -m src.parser.visualise_parsing path_to_document --show-report  # Processes the document as a report, shows dictionary output.
"""
import sys

from src.parser.report_processor import ReportProcessor
from src.parser.parsers import AutoParser

def main(argv: list[str] | None = None) -> int:
	""" Executes program. """
	argv = argv or sys.argv[1:]
	if not argv or len(argv) > 2:
		print(__doc__)
		return 2
	
	path = argv[0]
	
	if len(argv) == 1:
		parser = AutoParser(path)
		root = parser.parser.parse()
		root.print_tree()
		return 0
	
	if argv[1] != "--show-report":
		# Need to execute search 
		target = argv[1]
		root = root.label_search(target)
		if not root:
			print(f"Unable to find {target} in tree.")
			return 1
		root.print_tree()
		return 0 
	
	# Need to show report
	print(ReportProcessor(path).parse_tables()[0])
	return 0


if __name__ == "__main__":
	raise SystemExit(main())