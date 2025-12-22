""" Class for preprocessing the text present in trees. """
import re
from typing import Literal
from src.parser.trees import Tree
from src.extractor.material_collector import has_two_exams

class TreePreprocessor:
    allowed_latex = ["π"] # Fragmented LaTeX symbols which give context

    def __init__(
        self,
        doc_type: Literal["study_design", "past_exam"],
        remove_latex: bool = True,
    ):
        """ Initialises Tree Preprocessor object which is used for preprocessing
        trees parsed from exam documents. The types of documents supported
        for preprocessing are `past_exam` and `study_designs`.
        
        Parameters
        ----------
        doc_type    : Literal specifying whether the document is a study design or past exam. 
        remove_latex: Whether or not to skip all LaTeX-like fragments.
        """
        if doc_type not in ["study_design", "past_exam"]:
            raise ValueError("The only allowed doc types are 'study_design' and 'past_exam'.")
        self.doc_type = doc_type
        self.remove_latex = remove_latex        

    def preprocess(self, root: Tree,  **kwargs) -> Tree:
        """ Preprocesses trees. Call this method as `TextPreprocessor.preprocess(root, subject)`
        if initialised with `doc_type == "study_design"`. Otherwise, just the tree root is required.
        """
        if self.doc_type == "study_design":
            root = self._preprocess_sd(root, **kwargs)
        else:
            root = self._preprocess_exam(root)
        return root 

    def _preprocess_sd(self, root: Tree, subject: str) -> Tree:
        two_exams = has_two_exams(subject)
        if two_exams:
            root = root.label_search(rf"Units 3 and 4: {std_str(subject)}")[0]
        else:
            root.filter_tree(r"Unit(s)? (3|4)")
        aos_re = r"Area of Study \d?$"
        # only care about content-describing sections
        root.filter_tree(aos_re)
        if not two_exams:
            outcome_re = r"Outcome \d"
            level = root.find_node_level(outcome_re)
            outcome_nodes = root.label_search(outcome_re)
            # If outcomes aren't segmented into labelled topics and instead
            # into a key knowledge list, we have to find an appropriate label 
            # in other ways.  
            example_node = outcome_nodes[0] # Assume consistent document structure 
            if any([not re.match(r"Key ((K|knowledge)|(S|skills))", n.label)
                    for n in example_node.get_leaves()]):
                # leaf nodes already descriptive, don't need to do anything except collpase
                root.collapse(level=level, text_sep=". ")
                return root

            # No topic labels, check if parent label contains 
            # description (and not area of study text)
            if not re.match(aos_re, example_node.parent.label):
                root.filter_tree(outcome_re)
                root.collapse(level=level-1, label_sep=": ")
                return root
            
            # fallback: previous nodes with same parent give context
            for node in outcome_nodes:
                siblings = node.parent.children
                non_outcome_labels = [not re.match(outcome_re, sibling.label)
                                      for sibling in siblings]
                
                if any(non_outcome_labels):
                    context_label = siblings[non_outcome_labels.index(True)].label
                    node.label += ": " + context_label
                else:
                    # If all else fails, use area of study as label.
                    node.label = node.parent.label + ": " + node.label
            root.filter_tree(outcome_re)
            root.collapse(level=level, label_sep=": ")
            return root
        
        # Math study designs have a rigid structure, but inconsistent formatting
        # In particular, the first sub-heading under "Area of Study \d" gives a short description
        # of the Area of Study, then sub-sub-headings only, until the next area of study. 
        # We want to change this so the children of "Area of Study \d" are the sub-topics themselves. 
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

        # Collapse at topic description
        root.collapse(aos_level+1, text_sep=": ", label_sep=": ")
        return root 
    
    def _preprocess_exam(self, root: Tree) -> Tree:
        root.preprocess_text(self._preprocess_exam_text)
        return root 

    def _preprocess_exam_text(self, text: str) -> str:
        """ Processes input text by:
        1. removing LaTeX fragments (if initialised with `remove_latex=True`)
        2. Removing non-content related text
        """
        if self.remove_latex:
            text = self._remove_latex(text)
        
        text = re.sub(r"(?i)20\d\d .+ exam(ination)?", "", text) # remove headers
        text = re.sub(r"(\(?\d+\))? ?marks?", "", text) # remove leftover marks indicators
        text = re.sub(r"\(\d+\)*", "", text) # page numbers and stuff scattered about, remove them
        text = text.replace("  ", " ") # replace double spaces with just one space. 
        return text

    def _remove_latex(self, text: str) -> str:
        text = remove_pua_chars(text)
        # Remove every piece of text which is one character long in length. This removes a 
        # very large majority of LaTeX fragmentation with small false positive rate, given 
        # we include the letter "a" (not perfect, but fine).
        tokens = text.split(" ")
        keep = [t for t in tokens if t in self.allowed_latex or len(t) > 1 or t=="a"]
        return " ".join(keep)


def std_str(txt: str):
    """ Standardises capitalisation form snake_case to "Name Case". """
    s = txt.replace("_", " ")
    return " ".join([word[0].upper() + word[1:] for word in s.split()])


def remove_pua_chars(s: str) -> str:
    return ''.join(c for c in s if not (0xE000 <= ord(c) <= 0xF8FF))
