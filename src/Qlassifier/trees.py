from typing import List

class Tree: 
    def __init__(self, label: str, level: int):
        self.label = label # Tree title
        self.level = level # Distance from root
        self.text = '' # Text directly underneath header
        self.sub_headings = []
    

    def build(self, nodes: List["Tree"]):
        ''' Builds the tree from a list nodes. Nodes must be structured such that 
        each child of a node appears directly after it and before the next sibling node.

        Uses a stack-based algorithm.
        '''
        stack = [self]
        for node in nodes:
	    	# Pop until we find a parent of lower level
            while stack and stack[-1].level >= node.level:
                stack.pop()
            stack[-1].sub_headings.append(node)
            stack.append(node)


    def label_search(self, target: str) -> "Tree":
        ''' Searches tree for a child node with label==target using depth-first search (dfs). 
        Returns None if no match was found. 
        '''
        if self.label == target: return self

        for node in self.sub_headings:
            found = node.label_search(target)
            if found: return found

        return None
    

    def print_tree(self):
        ''' We implement a dfs to print the tree in a manner 
        which makes hierarchy clear. 
        ''' 
        indent = "  " * max(0, self.level-1)
        print(f"{indent} + [{self.level}]: {self.label}\n"
              f"Text: {self.text:>{len(indent)}}")
        for node in self.sub_headings:
            node.print_tree()
        