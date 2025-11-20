""" Object definitions for parsing word documents. 
"""
from docx import Document

from src.Qlassifier.trees import Tree


class WordParser:
	""" Word parser object. Supports the splitting of word documents into tree objects based on headers. """
	def __init__(self, path: str):
		""" Initialises word parsing object. Takes a path augment and loads in the word doc."""
		self.path = path
		self.doc = self._load_doc()

	def split_headings(self) -> Document: 
		""" Return a list of (level, text) for headings found in the document found at path. 
		It operates on the assumption that the document's headings are stylised as headings
		(as opposed to something like boldness/font size).
		"""
		headings = []
		# First, build the headings
		curr = None
		for p in self.doc.paragraphs:
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
				curr = Tree(label=text, level=level)
				headings.append(curr)
			elif curr: 
				curr.text += " " + text
		# Build the tree then return
		root = Tree(label="root", level=0)
		root.build(headings)
		return root

	def _load_doc(self):
		""" Loads the word document. """
		try:
			doc = Document(self.path)
		except Exception as e: 
			print(f"Error loading word document \'{self.path}\'. Path probably does not "
		 		  f"point to an existent file.")
			raise e
		return doc 


