""" Class for preprocessing the text present in trees. """
import re
from typing import Literal

from src.parser.trees import Tree

class TreePreprocessor:
    allowed_latex = ["π"] # Fragmented LaTeX symbols which give context

    def __init__(
        self,
        doc_type: Literal["study_design", "past_exam"],
        remove_latex: bool = True,
    ):
        """ Initialises Tree Preprocessor object which is used for preprocessing
        trees parsed from exam documents. The types of documents supported
        for preprocessing are past_exam and study_designs.

        remove_latex: Whether or not to skip all LaTeX-like fragments.
        is_math     : Does the document originate from a math subject 
        """
        if doc_type not in ["study_design", "past_exam"]:
            raise ValueError("The only allowed doc types are 'study_design' and 'past_exam'.")
        self.doc_type = doc_type
        self.remove_latex = remove_latex        

    def preprocess(self, root: Tree,  **kwargs) -> Tree:
        """ Preprocesses trees. Call this method as TextPreprocessor.preprocess(root, subject)
        if initialised with doc_type == "study_design". Otherwise, just the tree root is required.
        """
        if self.doc_type == "study_design":
            root = self._preprocess_sd(root, **kwargs)
        else:
            root = self._preprocess_exam(root)
        return root 

    def _preprocess_sd(self, root: Tree, subject: str) -> Tree:
        is_math = "math" in subject.lower()
        if is_math:
            root = root.label_search(rf"Units 3 and 4: {std_str(subject)}")
        else:
            root.filter_tree(r"Unit(s)? (3|4)")
        
        aos_re = r"Area of Study \d"
        # only care about content-describing sections
        root.filter_tree(aos_re)
        if not is_math:
            return root
        # Math study designs have a rigid structure, but inconsistent formatting
        # In particular, the first sub-heading under "Area of Study \d" give a short description
        # of the Area of Study, then sub-sub-headings only, until the next area of study. 
        
        aos_level = root.find_node_level(aos_re)
        aos_nodes = root.get_nodes_at_level(aos_level)

        for aos_node in aos_nodes:
            if len(aos_node.children) == 1:
                continue
            
            desc_node = aos_node.children[0] # Node containing text for AOS description
            new_desc_children = []
            for idx, node in enumerate(aos_node.children):
                if idx == 0:
                    continue
                # Remaining nodes are direct children of AOS node when they shouldnt be.
                node.level += 1
                new_desc_children.append(node)
                for child in node.children:
                    new_desc_children.append(child)
                node.children = []
            
            desc_node.children.extend(new_desc_children)
            aos_node.children = [desc_node]
            
        return root 
    
    def _preprocess_exam(self, root: Tree) -> Tree:
        root.preprocess_text(self._preprocess_exam_text)
        return root 

    def _preprocess_exam_text(self, text: str) -> str:
        """ Processes input text by:
        1. removing LaTeX fragments (if initialised with remove_latex=True)
        2. Removing non-content related text
        """
        if self.remove_latex:
            text = self._remove_latex(text)
        
        text = re.sub(r"(?i)20\d\d .+ exam(ination)?", "", text) # remove headers
        text = re.sub(r"(\(?\d+\))? ?marks?", "", text) # remove leftover marks indicators
        text = re.sub(r"\(\d+\)*", "", text) # page numbers and stuff scattered about, remove htem
        text = text.replace("  ", " ") # replace double spaces with just one space. 
        return text

    def _remove_latex(self, text: str) -> str:
        text = remove_pua_chars(text)
        # Simply remove every piece of text which 
        tokens = text.split(" ")
        keep = [t for t in tokens if t in self.allowed_latex or len(t) > 1]
        return " ".join(keep)


def std_str(txt: str):
    """ Standardises capitalisation form snake_case to "Name Case". """
    s = txt.replace("_", " ")
    return " ".join([word[0].upper() + word[1:] for word in s.split()])


def remove_pua_chars(s: str) -> str:
    return ''.join(c for c in s if not (0xE000 <= ord(c) <= 0xF8FF))