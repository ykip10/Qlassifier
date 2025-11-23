""" Contains definitions for tree structures which we use to store
most data. Trees support label search, filtering, printing, and other functions 
for processing data in the tree format. Node labels in the object definition 
are synonomous to the header titles from the documents from which the trees are built. 
"""
from typing import Self
import re


class Tree:
    """ Trees contain data on all processed documents in this project. 
    The attributes of a tree are:

    label   : Title of the node. if the node represents a question, this would include the question number
    level   : Distance from the root node
    children: Immediate child nodes
    parent  : Parent node
    text    : Text directly underneath the header, from whatever document the tree was processed from 
    marks   : If the node represents a question, this stores the number of marks available in the question. Otherwise, it is 0. 
    """
    def __init__(
            self, 
            label: str, 
            level: int,
            page_idx: int = None,
            ):
        """ Initialises Tree object. 

        label   : Title of the node.
        level   : Distance from the tree root. 
        page_idx: Which page the label was found (
        height  : Label height on page
        """
        self.label = label
        self.level = level
        self.children = []
        self.parent = None                               # Distance from root
        self.text = ""          # Text directly underneath header 
        self.marks = 0          # If the node references a question, stores the number of marks.
        self.has_mcq = self._has_mcq()    # Whether or not the tree has an mcq section. 

        # page num and height of label, if applicable
        self.page_idx = page_idx

    def build(self, nodes: list[Self]):
        """ Builds the tree from a list nodes. Nodes must be structured such that 
        each child of a node appears directly after it and before the next sibling node.

        Uses a simple stack-based algorithm.
        """
        stack = [self]
        for node in nodes:
	    	# Pop until we find a parent of lower level
            while stack and stack[-1].level >= node.level:
                stack.pop()
            node.parent = stack[-1]
            stack[-1].children.append(node)
            stack.append(node)

    def label_search(self, pattern: str) -> Self:
        """ Searches tree for a child node with a regex match between the pattern and the node label using depth-first search (dfs). 
        Returns None if no match was found. 
        """
        if re.match(pattern, self.label):
            return self

        for node in self.children:
            found = node.label_search(pattern)
            if found: 
                return found

        return None

    def print_tree(self, hide_text: bool = False):
        """ We implement a dfs to print the tree in a manner which makes hierarchy clear. """
        indent = "    " * max(0, self.level-1)
        print(f"{indent} [{self.level}]: {self.label}\n")
        if not hide_text:
            print(f"Text: {self.text:>{len(indent)}}"+ (f" ({self.marks} marks)" if self.marks else ""))
        
        for node in self.children:
            node.print_tree()

    def filter_tree(self, pattern: str) -> bool: 
        """ Goes through the tree and only keeps the branches which contain a node with a
        label that match the input regex pattern. Edits the tree in-place. 
        """
        keep = bool(re.search(pattern, self.label))
        if keep:
            return True
        
        # process children
        kept_children = []
        for child in self.children:
            if child.filter_tree(pattern):
                kept_children.append(child)

        # replace children list with only the kept ones
        self.children = kept_children

        # keep this node if it matches, or if any of it's children match. 
        return bool(self.children)

    def find_qn_level(self, qn_regex: str = r".?(Q|q)uestion.?") -> int:
        """ Finds the level in the tree which are labelled by the word question. """
        return self.label_search(qn_regex).level

    def get_nodes_at_level(self, level: int) -> list[Self]:
        """ Returns a list of nodes belonging to the specified level."""
        out = []
        # Implement breadth first search 
        queue = [self]
        while queue:
            node = queue.pop(0)
            for child in node.children:
                queue.append(child)
            if node.level == level:
                out.append(node)
            elif node.level > level: 
                return out
        return out
    
    def has_mcq(self):
        return bool(self.label_search(r"(i:)Section [a-zA-Z]"))
