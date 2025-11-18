''' Contains definitions for tree structures which we use to store
most data.
'''
from typing import List
import re

class Tree: 
    def __init__(self, label: str, level: int):
        self.label = label 
        self.level = level  # Distance from root
        self.text = ''      # Text directly underneath header
        self.marks = 0      # If the node is a question, stores the amount of marks in the question
        self.children = []
        self.parent = None


    def build(self, nodes: List["Tree"]):
        ''' Builds the tree from a list nodes. Nodes must be structured such that 
        each child of a node appears directly after it and before the next sibling node.

        Uses a simple stack-based algorithm.
        '''
        stack = [self]
        for node in nodes:
	    	# Pop until we find a parent of lower level
            while stack and stack[-1].level >= node.level:
                stack.pop()
            node.parent = stack[-1]
            stack[-1].children.append(node)
            stack.append(node)


    def label_search(self, target: str) -> "Tree":
        ''' Searches tree for a child node with label==target using depth-first search (dfs). 
        Returns None if no match was found. 
        '''
        if self.label == target: return self

        for node in self.children:
            found = node.label_search(target)
            if found: return found

        return None
    

    def print_tree(self, hide_text=False):
        ''' We implement a dfs to print the tree in a manner 
        which makes hierarchy clear. 
        ''' 
        indent = "    " * max(0, self.level-1)

        print(f"{indent} [{self.level}]: {self.label}\n")
        if not hide_text:
            (f"Text: {self.text:>{len(indent)}}"+ (f" ({self.marks} marks)" if self.marks else ""))
        
        for node in self.children:
            node.print_tree()


    def delete_node(self):
        ''' Deletes a node and all its subchildren from the tree. If called on the root, 
        function will not do anything. 
        '''
        if self.parent: 
            self.parent.children.remove(self)
    

    def filter_tree(self, pattern): 
        ''' Goes through the tree and only keeps the branches which contain a node with a
        label that match the input regex pattern. Uses bottom-up approach.
        '''
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
        