""" Contains definitions for tree structures which we use to store
most data. Trees support label search, filtering, printing, and other functions 
for processing data in the tree format. Node labels in the object definition 
are synonomous to the header titles from the documents from which the trees are built. 
"""
from typing import Self, Callable, Any
import re
import pandas as pd

class Tree:
    """ Trees are intermediate data structures used to store parsed data in this project. 
    The attributes of a tree are:

    `label`   : Title of the node. if the node represents a question, this would 
                include the question number
    `level`   : Distance from the root node
    `children`: Immediate child nodes
    `parent`  : Parent node
    `text`    : Text directly underneath the header, from whatever document 
                the tree was processed from 
    `marks`   : If the node represents a question, this stores the number 
                of marks available in the question. Otherwise, it is 0. 
    """
    def __init__(
        self, 
        label: str, 
        level: int,
        page_idx: int | None = None,
        subject_name: str | None = None,
        text: str | None = None,
        marks: int = 0,
    ):
        """ Initialises Tree object. 

        `label`       : Title of the node.
        `level`       : Distance from the tree root. 
        `page_idx`    : Which page the label was found
        `subject_name`: If applicable, the name of the subject w.r.t which the tree is relevant
        `text`        : Description of each node 
        `marks`       : If the node references a question, stores the number of marks.
        """
        self.label = label
        self.level = level
        self.page_idx = page_idx
        self.subject_name = subject_name if subject_name is not None else ""
        self.text = text if text is not None else ""
        self.marks = marks     
        
        self.parent = None
        self.children = []
        self.has_mcq = self._has_mcq()
    
    def __repr__(self): 
        return f"Tree(label={self.label!r}, level=({self.level!r}))"

    def _has_mcq(self):
        return bool(self.label_search(pattern=r"(?i)multiple[- ]choice"))

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

    def label_search(self, pattern: str, non_re: Any | None = None) -> list:
        """ Searches tree for a child node with a regex match between the pattern and the node 
        label using depth-first search (dfs). Returns None if no match was found. 
        """
        matches = []
        stack = [self]
        while stack:
            node = stack.pop()
            if re.match(pattern, node.label):
                matches.append(node)
            stack.extend(node.children)
        return matches

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
        keep = bool(re.match(pattern, self.label))
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

    def find_node_level(self, label_regex: str = r"Question") -> int:
        """ Finds the level in the tree which are labelled by the input label regex. 
        Useful when an entire level shares a regex naming pattern 
        (such as Question 1, 2...)
        """
        return self.label_search(label_regex)[0].level

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

    def collapse(
        self,
        level: int,
        text_sep: str | None = None,
        label_sep: str | None = None,
    ):
        """ Recursively collapses tree at specified level, concatenating
        parent text and labels with child text and labels.
        
        Example Tree:
            Level 2: [Q1]
            Level 3: [a,b,c]

            After collapse, we have

            Level 2: [Q1a, Q2b, Q2c] with text concatenated. 
            Level 3: Empty
        
        Args:
            level         : level to collapse at 
            text_sep      : Character to use as a separator between text when concatenating. If `None`, 
                            defaults to empty string
            label_sep     : Character to use as a separator between labels when concatenating. If `None`,
                            doesn't concatenate labels, just uses child label
        """
        child_label = label_sep is None # Whether or not we should use child labels
        label_sep = label_sep if label_sep is not None else ""
        text_sep = text_sep if text_sep is not None else ""

        # Approach: iterate over all nodes at the collapsing level 
        # and replace their parent's children with the collpased nodes
        nodes = self.get_nodes_at_level(level)
        for node in nodes:
            if not node.children:
                # already collapsed
                continue
            parent = node.parent

            collapsed = [] # collapsed nodes
            self._dfs_collect(
                node, "", "", collapsed, node.level, text_sep, label_sep, child_label
            )
            
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
        text_sep: str,
        label_sep: str,
        child_label: bool
    ):
        """ Collapses all nodes from input node, edits the out list to contain
        the flattened collapsed, nodes. 
        """
        new_label = label_prefix + label_sep + node.label if not child_label else node.label
        new_text = text_prefix + text_sep + node.text 

        if not node.children:
            # Labels and text have extra separators at the end, just shave that off
            new_label = new_label[len(label_sep):]
            new_text = new_text[len(text_sep):]
            new = Tree(
                label=new_label,
                level=level,
                text=new_text,
                page_idx=node.page_idx,
                marks=node.marks
            )
            out.append(new)
            return

        for child in node.children:
            child._dfs_collect(
                child, new_label, new_text, out, level, text_sep, label_sep, child_label
            )
    
    def to_df(self, include_root: bool = True, level = 0):
        """ Converts the tree to a pandas DataFrame with columns:
        `label`: node label
        `text` : associated text for the node

        If `level==0`, 
        Traverses the tree in depth-first order and optionally includes the root node.

        If `level>0`, only converts the nodes at level into a df.
        """
        if level: 
            nodes = self.get_nodes_at_level(level)
            df = pd.DataFrame(
                data={
                    "label": [node.label.strip() for node in nodes],
                    "text": [node.text.strip(" :\n\t,") for node in nodes]
                }
            )
            return df
        
        rows = []
        def _dfs(node):
            rows.append({"label": node.label, "text": node.text})
            for child in node.children:
                _dfs(child)

        _dfs(self)
        df = pd.DataFrame(rows, columns=["label", "text"])
        if not include_root:
            df = df.loc[df["label"] != self.label, :]
        return df
    
    def edit_node(self, new_node: Self) -> Self:
        """Replace this node with new_node in the tree.
        Requires the input node to be of the same level as the 
        current node. 
        """
        if new_node.level != self.level: 
            raise ValueError("New nodes level must be the same as current node.")

        # detach new_node from its current parent (if any)
        if new_node.parent is not None:
            try:
                new_node.parent.children.remove(new_node)
            except ValueError:
                pass
            new_node.parent = None

        parent = self.parent
        if parent is None:
            # replacing the root, mutate self to adopt new_node's attributes and children
            self.label = new_node.label
            self.page_idx = new_node.page_idx
            self.subject_name = new_node.subject_name
            self.text = new_node.text
            self.marks = new_node.marks

            # Use only new_node's children
            self.children = []
            for ch in new_node.children:
                ch.parent = self
                self.children.append(ch)
            self.has_mcq = self._has_mcq()
            return self

        # replace self in parent's children with new_node
        try:
            idx = parent.children.index(self)
        except ValueError:
            # fallback: append if self is not found
            parent.children.append(new_node)
        else:
            parent.children[idx] = new_node

        new_node.parent = parent
        # update has_mcq on the subtree root
        new_node.has_mcq = new_node._has_mcq()
        return new_node

    def get_leaves(self) -> list[Self]:
        """Return a list of leaf nodes in this subtree.
        Traversal is depth-first and includes `self` if it is a leaf.
        """
        leaves = []
        def _dfs(node: Self):
            if not node.children:
                leaves.append(node)
                return
            for child in node.children:
                _dfs(child)

        _dfs(self)
        return leaves

