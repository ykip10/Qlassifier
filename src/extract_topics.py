''' Tools for processing word documents into trees by splitting on headers. 
'''

import sys
from typing import List, Tuple
from docx import Document

# We store our headings in an N-ary tree structure, built in order 
# of occurence in the input word document. 
class Heading: 
	def __init__(self, title: str, level: int):
		self.title = title # Header title 
		self.level = level # Distance from root
		self.text = '' # Text directly underneath header
		self.sub_headings = []


def extract_headings(path: str) -> List[Tuple[int, str]]:
	''' Return a list of (level, text) for headings found in the document. 
	It operates on the assumption that the document's headings are stylised as headings
	(as opposed to something like boldness/font size).
	'''
	try:
		doc = Document(path)
	except Exception: 
		print(f"Error loading word document \'{path}\'")
		return []
	
	headings = []
	root = Heading("root", level=0)

	# First build the headings
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
			curr = Heading(text, level)
			headings.append(curr)
		else: 
			if curr: 
				curr.text += text + " "

	# Now we sort out the hierarchical tree structure
	def add_children(root, headings):
		''' Finds the subchildren for each node and organises headings list 
		into a tree structure using a stack-based algorithm. 
		'''
		stack = [root]
		for node in headings:
			# Pop until we find a parent of lower level
			while stack and stack[-1].level >= node.level:
				stack.pop()
			stack[-1].sub_headings.append(node)
			stack.append(node)

	add_children(root, headings)
	return root


def find_heading(root, title):
	''' Recursively performs a dfs on the root to search for a node with title==title. 
	Note this returns the first node traversed which satisfies this condition, which may 
	or may not be unique. 
	'''
	if root.title == title:
		return root 
	
	for node in root.sub_headings:
		found = find_heading(node, title)
		if found:
			return found

	return None


def print_tree(root):
	''' We implement a depth-first search to print the tree in a manner 
	which makes hierarchy clear. 
	''' 
	indent = "  " * max(0, root.level-1)
	print(f"{indent} + [{root.level}]: {root.title}\n"
		  f"Text: {root.text:>{len(indent)}}")
	for node in root.sub_headings:
		print_tree(node)


def main(argv: List[str] | None = None) -> int:
	argv = argv or sys.argv[1:]
	if not argv and len(argv) != 2:
		print("Usage: python process_doc.py path_to_word_doc.docx search")
		return 2
	
	path = argv[0]
	search = argv[1]
	headings = extract_headings(path)

	res = find_heading(headings, search)
	if not res:
		print(f"Unable to find {search} in tree.")
		return 1
	print_tree(res)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
