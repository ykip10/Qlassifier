""" Includes logic and object definitions for the report parser. This parser has a different goal to 
the word and PDF parsers, which each would like to parse a document into a Tree object. The report parser,
which allows report input as either PDF or word, aims to find each question discussed and output a dictionary 
outlining it's correct answer, any comments, and the mark distribution associated with it (% of students who got X marks)
"""
import re
from pathlib import Path
from copy import deepcopy
from collections import defaultdict as dd

from pprint import pprint
from docx.enum.text import WD_COLOR_INDEX
import pandas as pd

from src.Qlassifier.parsers import PDFParser, WordParser
from src.Qlassifier import pdf_utils

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
		self.path = path
		self.parser = WordParser(path) if Path(path).suffix == ".docx" \
					  else PDFParser(path, (0.07, 0.06, 0.93, 0.93))
		self._results = []

		# tree representation of report
		self._root = self.parser.split_headings()
		# question labels for short answer questions e.g. ['Question 1ai', 'Question 1aii', ... ]
		self.sa_qns = iter(self.get_sa_qns()) 

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
		""" We extract n answers from reports and their associated mark distributions in this function.  
		Parses word tables where, in the case of multiple choice questions:
		  - first column is question identifier
		  - last column is comments
		  - middle columns are percentages for each option
			(the correct answer is indicated by highlighted text in the cell)
		
		Returns a list of dicts:
			  	[{'question': '1', 
		    	 'options': [{'index': 0, 'percentage': 45.0, 'is_correct': True}, ...],
				 'comments': '...'}, 
		   		 ...]
				   
		The dicts are of different structure depending on whether or not the question processed is short-answer.
		The above is an example of a dict constructed from a multiple choice question (hence the is_correct flag).
		"""
		if isinstance(self.parser, WordParser):
			return self.parse_word_tables()
		elif isinstance(self.parser, PDFParser):
			return self._parse_pdf_tables()
		return []	

	def _parse_word_tables(self):
		""" See parse_tables. This function assumes the report being parsed is a word doc."""
		results = []
		for table in self.parser.doc.tables:
			# as long as there are more than 2 rows in the table, its
			top_left_text = table.rows[0].cells[0].text.strip().lower()
			mcq = len(table.rows) > 2 or top_left_text == "question" \
				  or top_left_text == "" 
			if mcq: 
				if not self._parse_word_table(table, curr=results, mcq=True):
					print("Error parsing mcq questions from report.")
					return 
			elif top_left_text == "marks":
				# We are now short in the answer portion of the report
				if not self._parse_word_table(table, curr=results, mcq=False):
					print("Error parsing short-answer questions from report.")
					return
			else:
				# not a table containing marks
				continue
		return results
				
	def _parse_word_table(
			self,
			table: object,
			curr: list[pd.DataFrame],
			mcq: bool = True,
		) -> int:
		""" Parses a singular word table. """
		# Extract column names
		cols = []
		header = table.rows[0]
		for cell in header.cells:
			cols.append(cell.text.strip())

		table_data = dd(list)
		if mcq: 
			cols.append("is_correct")
		else:
			# Short answer, need to extract comments 
			# from outside the table. 
			cols.append("comments")
			comment = self.root.label_search(next(self.sa_qns)).text
			table_data["comments"] = comment
		
		for row in table.rows[1:]:
			cells = row.cells
			# first column not a question column
			table_data[cols[0]].append(cells[0].text.strip())
			qn_cells = cells[1:-1]
			if mcq:
				# second to last column contains comments
				table_data[cols[-2]].append(cells[-1].text.strip())
		
			for idx, cell in enumerate(qn_cells):
				table_data[cols[idx+1]].append(cell.text.strip())
				if mcq and cell_highlighted(cell):
					table_data["is_correct"].append(idx)
		
		curr.append(pd.DataFrame(columns=cols, data=table_data))
		return 1

	def _parse_pdf_tables(self):
		""" See parse_tables. This function assumes the report being parsed is a PDF.

		Strategy: 
		- Use table extraction methods on Section A (mcq) questions. Loses information 
		  on the correct answer since we cannot reliably preserve highlighting.

		- On section B, we use the raw text. We look for strings of numbers which 
		  are at most one-off in their running sum from 100%, and assume these to correspond  
		  to the mark mass of each question.
		"""
		# extract all tables inside the pdf
		results = pdf_utils.get_tables(self.path)

		# Still need to get the comments outside of tables
		root = self.root
		comments = pd.DataFrame(
			data={"comments": [root.label_search(qn).text for qn in self.sa_qns]}
			)
		results = self._arrange_tables(results)
		for table in results[1:]: # exclude joined mcq table
			table = pd.concat([table, comments], axis=1)
		return results
	
	@staticmethod
	def _arrange_tables(dfs: list[pd.DataFrame]) -> list[pd.DataFrame]:
		""" Merges list of dfs results so that all tables with comments 
		are grouped together. If no tables with comments (no mcq questions)
		then leaves dfs unmodified.
		"""
		# want to avoid iterative concatenating,
		# find end of mcq section
		sa_start_idx = 0 
		for df in dfs:
			if "comments" in [col.lower() for col in df.columns]:
				sa_start_idx += 1
			else: 
				break
		# edit list to have the first entry as the merged df,
		# all other entries individual tables
		if sa_start_idx:
			merged = pd.concat(dfs[:sa_start_idx], axis=0)
			dfs = dfs[(sa_start_idx-1):]
			dfs[sa_start_idx-1] = merged
		return dfs

