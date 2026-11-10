"""HELD-OUT test: a different domain than specific_eval.py, written after the matching rules
were tuned, with paraphrased questions. Run: python scripts/heldout_eval.py [--verbose]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scripts.specific_eval as ev  # noqa: E402

PAGES = [
"""CS201 Data Structures: Course Syllabus

Course overview
CS201 is taught by Professor Lakshmi Varma at Greenfield University. Lectures are held on Tuesdays and Thursdays from 14:00 to 15:30 in Room B-204. Teaching assistant Rohan Kulkarni holds office hours every Wednesday afternoon.

Prerequisites
Students must have passed CS101 Programming Fundamentals with a grade of at least C. Familiarity with Python is assumed throughout the semester.""",
"""Grading

The final grade is computed from four components. Weekly assignments contribute 30 percent, the midterm exam contributes 25 percent, the final exam contributes 35 percent, and class participation makes up the remaining 10 percent. The lowest assignment score is dropped automatically.

Late submissions lose 10 percent of the marks per day, and work more than 5 days late is not accepted. Requests for extensions must be sent to the teaching assistant before the deadline.""",
"""Schedule and topics

The semester has 14 weeks. Weeks 1 to 3 cover arrays and linked lists. Weeks 4 to 6 introduce stacks, queues and recursion. The midterm exam takes place in week 7 on 18 March. Trees and heaps are taught in weeks 8 to 10, followed by hash tables and graphs in weeks 11 to 13. Week 14 is reserved for revision.

Textbook
The recommended textbook is Introduction to Algorithms by Cormen, Leiserson, Rivest and Stein. Copies are available in the library, but each student may borrow a copy for at most 2 weeks.""",
"""Academic integrity

Copying code from classmates or online sources without attribution is treated as plagiarism. A first offence results in a zero for the assignment, and a second offence is reported to the disciplinary committee. Students are encouraged to discuss ideas, but every submission must be written individually.

Accommodations
Students with documented disabilities can request extra exam time of up to 50 percent by contacting the Accessibility Office at least 3 weeks before the exam.""",
]
QA = [
    ("Which professor teaches this course?", "lakshmi varma", 1),
    ("What room are lectures in?", "b-204", 1),
    ("At what time do classes start?", "14:00", 1),
    ("When can I meet the TA?", "wednesday", 1),
    ("What grade do I need in CS101 to enrol?", "at least c", 1),
    ("What percentage of the grade comes from the final exam?", "35", 2),
    ("How much is participation worth?", "10 percent", 2),
    ("What happens to my lowest assignment mark?", "dropped", 2),
    ("What is the penalty for handing in work late?", "10 percent", 2),
    ("After how many days are late assignments refused?", "5 days", 2),
    ("Who approves deadline extensions?", "teaching assistant", 2),
    ("How many weeks long is the term?", "14 weeks", 3),
    ("When is the midterm?", "18 march", 3),
    ("Which topics are covered in weeks 8 to 10?", "trees and heaps", 3),
    ("What book is recommended?", "introduction to algorithms", 3),
    ("For how long can the textbook be borrowed?", "2 weeks", 3),
    ("What is the consequence of a second plagiarism offence?", "disciplinary", 4),
    ("How much extra time can students with disabilities get?", "50 percent", 4),
    ("How early must I contact the Accessibility Office?", "3 weeks", 4),
    # unanswerable look-alikes
    ("What is the tuition fee for CS201?", None, None),
    ("Who is the head of the university?", None, None),
    ("How many students are enrolled in the course?", None, None),
    ("Is there a project in the course?", None, None),
]
if __name__ == "__main__":
    ev.PAGES, ev.QA = PAGES, QA          # reuse the runner from specific_eval.py
    ev.main(False, "--verbose" in sys.argv)
