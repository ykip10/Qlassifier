import re

from docx.enum.text import WD_COLOR_INDEX

from src.Qlassifier.pdf_parser import PDFParser
from src.Qlassifier.word_parser import WordParser


class ReportProcessor:
	""" Processor containing methods for processing VCAA documents as either PDFs or Word docs."""
	def __init__(self, path: str, ext: str):
		""" Creates an object for report processing. 	
		path: Report path
		ext : File extension (.pdf or .docx)
		"""
		self.path = path
		self.parser = WordParser(path) if ext == ".docx" else PDFParser(path)
		self.results = []

		# question labels for short answer questions e.g. ['Question 1ai', 'Question 1aii', ... ]
		self.sa_qns = iter(self._get_sa_qns()) 

	def parse_tables(self) -> list[dict]:
		""" We extract n answers from reports and their associated mark distributions in this function.  
		Parses word tables where, in the case of multiple choice questions:
		  - first column is question identifier
		  - last column is comments
		  - middle columns are percentages for each option
			(the correct answer is indicated by highlighted text in the cell)
		
		Returns a list of dicts:
			  	[{'question': '1', 
		    	 'options': [{'index': 0, 'text': 'A 45%', 'percentage': 45.0, 'is_correct': True}, ...],
				 'comments': '...'}, 
		   		 ...]
				   
		The dicts are of different structure depending on whether or not the question processed is short-answer.
		The above is an example of a dict constructed from a multiple choice question (hence the is_correct flag).
		"""
		if isinstance(self.parser, WordParser):
			self._parse_word_tables()
			return self.results
		elif isinstance(self.parser, PDFParser):
			self._parse_pdf_tables()
			return self.results
		return []	

	def _parse_word_tables(self):
		""" See parse_tables. This function assumes the report being parsed is a word doc."""
		pct_re = re.compile(r'\s?\d{1,2}\s?') # regex for 2 digit numbers
		for table in self.parser.doc.tables:
			if table.rows[0].cells[0].text.strip().lower() == "question":
				if not self._parse_word_mcq_table(table, pct_re):
					print("Error parsing mcq questions from report.")
					return 
			else:
				# We are now short in the answer portion of the report
				if not self._parse_word_sa_table(table, pct_re):
					print("Error parsing short-answer questions from report.")
					return 
				
	def _parse_word_mcq_table(self, table: object, pct_re: str) -> int:
		""" Parses a row of a table present in a VCAA report with the following structure
		| Question | % A | % B | % C | ... | Comments | where the correct
		answer is highlighted into a dictionary.
		"""
		for row in table.rows:
			cells = row.cells
			if len(cells) < 3:
				return 0 # need at least question, one option, comments
			
			# First and last rows contain question id and comments, respectively
			question_id = cells[0].text.strip()
			comments = cells[-1].text.strip()

			option_cells = cells[1:-1]
			options = []
			for idx, cell in enumerate(option_cells):
				cell_text = cell.text.strip()
				m = pct_re.search(cell_text)
				percentage = float(m.group(0)) if m else None
		
				# Detect highlight on any run in the cell
				is_highlighted = False
				for para in cell.paragraphs:
					for run in para.runs:
						try:
							hc = run.font.highlight_color
						except Exception:
							hc = None
						if hc is not None and hc != WD_COLOR_INDEX:
							is_highlighted = True
							break
					if is_highlighted:
						break
					
				options.append({
					"index": idx,
					"text": cell_text,
					"percentage": percentage,
					"is_correct": is_highlighted,
				})
		
			self.results.append({
				"question": question_id,
				"options": options,
				"comments": comments,
			})
		return 1

	def _parse_word_sa_table(self, table: object, pct_re: str) -> int:
		""" Handles the short-answer case. Parses a table which is assumed to contain the mark distribution of a 
		question with columns [...| 0M % | 1M % | ... |...] and only two rows, where the first does not contain any data. 
		"""
		try:
			question_id = next(self.sa_qns)
		except StopIteration:
			# Done
			return 0

		root = self.parser.split_headings()
		comments = root.label_search(question_id).text # comments are underneath the relevant question header

		row = table.rows[1] # second row contains the data we want
		cells = row.cells
		if len(cells) < 3:
			raise Exception("Encountered an unexpected table.")
		
		marks = cells[1:-1] # marks are in the middle columns
		mark_dist = []
		for idx, cell in enumerate(marks):
			cell_text = cell.text
			m = pct_re.search(cell_text)
			percentage = float(m.group(0)) if m else None
			mark_dist.append({"marks": idx, "percentage": percentage})
		
		self.results.append({
			"question": question_id, 
			"marks_dist": mark_dist, 
			"comments": comments
		})
		return 1

	def _get_sa_qns(self):
		""" Finds the names of the short answer questions. """
		root = self.parser.split_headings()
		qn_regex = r"(Q|q)uestion \d.?"
		root.filter_tree(pattern=qn_regex)
		qn_level = root.find_qn_level(qn_regex=qn_regex)
		return [node.label for node in root.get_nodes_at_level(qn_level)]
		