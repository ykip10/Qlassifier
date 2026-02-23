"""
Qlassifier is not intended to support every VCE subject. It is for subjects
with text-based exam components which have common exam format between ALL students
(non-dynamic study-designs), and where the FULL CONTEXT
is provided in the questions themselves (excludes many humanities subjects).
This mostly amounts to STEM subjects.

The scope of subjects for which explicit support is provided (cached results,
more advanced models trained on hand-labelled data) is below. Other subjects
will have worse performance or not work at all.
"""

SUBJECTS = [
    "Foundation Mathematics",
    "General Mathematics",
    "Mathematical Methods",
    "Specialist Mathematics",
    "Biology",
    "Chemistry",
    "Environmental Science",
    "Physics",
    "Psychology",
    "Accounting",
    "Business Management",
    "Economics",
    "Legal Studies",
]

YEARS = range(2017, 2024)

