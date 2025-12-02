""" Contains definitions for tree structures which we use to store
most data. Trees support label search, filtering, printing, and other functions 
for processing data in the tree format. Node labels in the object definition 
are synonomous to the header titles from the documents from which the trees are built. 
"""
from typing import Self, Callable
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
        text: str = "",
        marks: int = 0
    ):
        """ Initialises Tree object. 

        label   : Title of the node.
        level   : Distance from the tree root. 
        page_idx: Which page the label was found (
        height  : Label height on page
        """
        self.label = label
        self.level = level
        self.visited = False
        self.children = []
        self.parent = None      # Distance from root
        self.text = text         # Text directly underneath header 
        self.marks = marks         # If the node references a question, stores the number of marks.
        self.has_mcq = self._has_mcq()    # Whether or not the tree has an mcq section. 

        # page num,if applicable
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
        """ Searches tree for a child node with a regex match between the pattern and the node 
        label using depth-first search (dfs). Returns None if no match was found. 
        """
        if re.search(pattern, self.label):
            return self

        for node in self.children:
            found = node.label_search(pattern)
            if found is not None: 
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

    def find_node_level(self, label_regex: str = r".?(Q|q)uestion.?") -> int:
        """ Finds the level in the tree which are labelled by the input label regex. 
        Useful when an entire level shares a regex naming pattern 
        (such as Question 1, 2...)
        """
        return self.label_search(label_regex).level

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
    
    def preprocess_text(self, preprocessor: Callable[[str], str]):
        """ Preprocesses all text in every child node with the preprocessor (function)."""
        if self.text:
            self.text = preprocessor(self.text)

        for node in self.children:
            node.preprocess_text(preprocessor)

    def collapse(self, level: int, sep: str = "", concat_label: bool = False):
        """ Recursively collapses tree at specified level, concatenating
        parent text and labels with child text and labels.
        
        Example Tree:
            Level 2: [Q1]
            Level 3: [a,b,c]

            After collapse, we have

            Level 2: [Q1a, Q2b, Q2c] with text concatenated. 
            Level 3: Empty
        
        Args:
        level       : level to collapse at 
        sep         : Character to use as a separator between labels when concatenating
        concat_label: Boolean flag indicating whether or not we should concatenate labels
                      (as opposed to just using the child's label)
        """

        # Approach: iterate over all nodes at the collapsing level 
        # and replace their parent's children with the collpased nodes
        nodes = self.get_nodes_at_level(level)
        for node in nodes:
            if not node.children:
                # already collapsed
                continue
            parent = node.parent

            collapsed = [] # collapsed nodes
            self._dfs_collect(node, "", "", collapsed, level=node.level,
                              sep=sep, concat_label=concat_label)
            
            # replace node with collapsed
            parent.children.remove(node)
            parent.children.extend(collapsed)

    def _dfs_collect(
        self,
        node: Self,
        label_prefix: str,
        text_prefix: str,
        out: list[Self],
        level: int,
        sep: str,
        concat_label: bool
    ):
        """ Collapses all nodes from input node, edits the out list to contain
        the flattened collapsed, nodes. 
        """
        new_label = label_prefix + sep + node.label \
                    if label_prefix and not concat_label else node.label 
        new_text = text_prefix + sep + node.text

        if not node.children:
            new = Tree(
                label=new_label,
                level=level,
                text=new_text,
                page_idx=node.page_idx,
                marks=node.marks
            )
            new.marks
            out.append(new)
            return

        for child in node.children:
            child._dfs_collect(child, new_label, new_text, out, level, sep, concat_label)
            
    def _has_mcq(self):
        return bool(self.label_search(r"(i:)Section [a-zA-Z]"))
