""" Includes logic and object definitions for the report parser. This parser has a different goal to 
the word and PDF parsers, which each would like to parse a document into a Tree object. The report parser,
which allows report input as either PDF or word, aims to find each question discussed and output a dictionary 
outlining it's correct answer, any comments, and the mark distribution associated with it (% of students who got X marks).
"""
from pathlib import Path
from copy import deepcopy
from collections import defaultdict as dd

from docx.enum.text import WD_COLOR_INDEX
import pandas as pd

from src.Qlassifier.parsers import PDFParser, WordParser
from src.Qlassifier.pdf_utils import get_tables, process_tables, convert_extracted_tables


def cell_highlighted(cell: object) -> bool:
    """ Helper functions. Checks if a docx cell is highlighted. """
    for para in cell.paragraphs:
        for run in para.runs:
            try:
                hc = run.font.highlight_color
            except Exception:
                hc = None
            if hc is not None and hc != WD_COLOR_INDEX:
                return True
    return False


class ReportProcessor:
	""" Processor containing methods for processing VCAA documents as either PDFs or Word docs."""
	def __init__(self, path: str):
		""" Creates an object for report processing. 	
		path: Report path
		ext : File extension (.pdf or .docx)
		"""
		self.path = str(path)
		self.parser = WordParser(path) if Path(path).suffix == ".docx" \
					  else PDFParser(path, (0.07, 0.06, 0.93, 0.93))
		self._results = []

		# tree representation of report
		self._root = self.parser.split_headings()
		# question labels for short answer questions e.g. ['Question 1ai', 'Question 1aii', ... ]
		self.sa_qns_lst = self.get_sa_qns()
		self.sa_qns = iter(self.sa_qns_lst) 

	@property
	def root(self):
		return deepcopy(self._root)

	def get_sa_qns(self) -> list[str]:
		""" Scrapes the names of the short answer questions from the report. """
		root = self.root
		qn_regex = r"(Q|q)uestion \d.?"
		root.filter_tree(pattern=qn_regex)
		qn_level = root.find_qn_level(qn_regex=qn_regex)
		return [node.label for node in root.get_nodes_at_level(qn_level)]

	def parse_tables(self) -> list[pd.DataFrame]:
		""" We extract answers from reports and their associated mark distributions in this function.  
		Parses PDF or word tables where, in the case of multiple choice questions:
		  - first column is question identifier
		  - last column is comments
		  - middle columns are percentages for each option
			(the correct answer is indicated by highlighted text in the cell)
		
		Returns a list of pandas dataframes, the first index of which is for the MCQ section,
		and one for the mark distribution of each short answer question. If the report is a PDF, 
		does not give data on which MCQ questions are correct. 
		"""
		if isinstance(self.parser, WordParser):
			return self._parse_word_tables()
		elif isinstance(self.parser, PDFParser):
			return self._parse_pdf_tables()
		return []	

	def _parse_word_tables(self):
		""" See parse_tables. This function assumes the report being parsed is a word doc."""
		mcq_dfs = []
		sa_dfs = []
		for table in self.parser.doc.tables:
			# as long as there are more than 2 rows in the table
			top_left_text = table.rows[0].cells[0].text.strip().lower()
			mcq = len(table.rows) > 2 or top_left_text == "question" \
				  or top_left_text == "" 
			if mcq: 
				if not self._parse_word_table(table, curr=mcq_dfs, mcq=True):
					print("Error parsing mcq questions from report.")
					return 
			elif "mark" in top_left_text:
				# We in the short answer portion of the report
				if not self._parse_word_table(table, curr=sa_dfs, mcq=False):
					print("Error parsing short-answer questions from report.")
					return
			else:
				# not a table containing marks
				continue
		results = mcq_dfs + sa_dfs
		return results
				
	def _parse_word_table(
		self,
		table: object,
		curr: list[pd.DataFrame],
		mcq: bool = True,
	) -> int:
		""" Parses a singular word table. Checks for correct answer column, if none are found
		looks for cell highlighting to extract correct answers. """
		# Extract column names
		cols = []
		header = table.rows[0]
		for cell in header.cells:
			cols.append(cell.text.strip())
		
		if len(cols) < 4:
			# skip this table, needs at least 4 columns to make sense
			return 1 
		
		# If we don't already have a correct answer column, add it
		cols_lower = [col.lower().strip() for col in cols]
		has_correct_ans_col = any("correct" in col for col in cols_lower)
		
		table_data = dd(list)
		if mcq and not has_correct_ans_col: 
			cols.append("is_correct")
		elif not mcq:
			# Short answer, need to extract comments 
			# from outside the table. 
			cols.append("comments")
			comment = self.root.label_search(next(self.sa_qns)).text
			table_data["comments"] = comment
		
		for row in table.rows[1:]:
			# Fill table by looking at cells 
			for idx, cell in enumerate(row.cells):
				table_data[cols[idx]].append(cell.text.strip())
				# may need to manually find correct answer
				if not has_correct_ans_col and mcq and cell_highlighted(cell):
					table_data["is_correct"].append(idx)
		
		curr.append(pd.DataFrame(columns=cols, data=table_data))
		return 1

	def _parse_pdf_tables(self):
		""" See parse_tables. This function assumes the report being parsed is a PDF. Since 
		we cannot reliably extract highlighting from PDFs, we do not gain information
		on which MCQ was correct (not true for word docs)
		"""
		# extract all tables inside the pdf
		extracted_tables = get_tables(self.path)
		mcq_dfs, sa_dfs = convert_extracted_tables(extracted_tables)
		results = process_tables(mcq_dfs, sa_dfs)

		# Still need to get the comments outside of tables
		root = self.root
		comment_dfs = [
			pd.DataFrame(data={"comments": [root.label_search(qn).text]})
			for qn in self.sa_qns_lst
		]

		for idx, df in enumerate(results):
			if idx == 0: # skip mcq table 
				continue
			try: 
				new_df = pd.concat(
					[df, comment_dfs[idx-1]],
					axis=1
				)
			except IndexError as IE:
				raise IE("Too many tables scanned were classified as non-mcq." \
						 "Fault is likely with table processing logic.")
			results[idx] = new_df
		return results
	