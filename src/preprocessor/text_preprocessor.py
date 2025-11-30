import re
from abc import ABC, abstractmethod

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from parser.trees import Tree

class TextPreprocessor:
    def __init__(
        self,
#        remove_latex: bool = True,
        remove_stopwords: bool = True,
        keep_alnum_only: bool = True
    ):
#        self.remove_latex = remove_latex
        self.remove_stopwords = remove_stopwords
        self.keep_alnum_only = keep_alnum_only
        self._stopwords = stopwords

    def _filter_stopwords(self, tokens: list) -> list:
        filtered = []
        for t in tokens:
            low = t.lower()
            # remove pure punctuation tokens and stopwords
            if low in self._stopwords:
                continue
            # keep tokens that contain at least one alphanumeric char
            if re.search(r'\w', t):
                if self._strip_non_alnum:
                    t = "".join([c for c in t if c.isalnum()])
                filtered.append(t)
        return filtered

    def preprocess(self, text: str) -> str:
        """
        Preprocess a single text string.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a str")

        # not tokenizing: optionally remove stopwords by tokenizing and joining back
        if self.remove_stopwords:
            nltk.download('stopwords', quiet=True)
            tokens = word_tokenize(text)
            tokens = self._filter_stopwords(tokens)
            return " ".join(tokens)

        # default: cleaned string
        return text


class TreePreProcessor:
    def __init__(self, textpreprocessor: TextPreprocessor):
        self.preprocessor = textpreprocessor

    def preprocess(self, root: Tree):
        # to be implemented
        pass
