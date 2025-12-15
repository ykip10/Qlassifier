import re

def normalise_question_label(s: str) -> str:
    """ Normalises question labels which become like Question 1.\tc\ti after collapsing.
    Normalises to be of the form Question 1ci. instead. 
    """
    pref = re.search(r"Question \d{1,2}", s).group()
    suff = "".join(c for c in s.replace(pref, "") if c.isalnum())
    return pref + suff + "."