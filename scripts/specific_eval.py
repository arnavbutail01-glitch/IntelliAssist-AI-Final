"""Harder evaluation: specific factual questions (who / when / how many / why / how)
over a multi-page document, including paraphrased wording and unanswerable
look-alike questions.   Run: python scripts/specific_eval.py [--hf] [--verbose]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.document_loader import Document  # noqa: E402
from core.embeddings import get_embedder  # noqa: E402
from core.rag_pipeline import answer_question, build_knowledge_base  # noqa: E402
from core.vector_store import VectorStore  # noqa: E402

PAGES = [
"""Northwind Analytics Employee Handbook

About the company
Northwind Analytics was founded in 2014 by Meera Iyer and Daniel Brooks in Pune, India. The company builds forecasting software for retailers. Its headquarters moved to Bengaluru in 2019, and the Pune office now serves as the support centre.

Leadership
The Chief Executive Officer is Meera Iyer. Daniel Brooks serves as Chief Technology Officer. Priya Nair leads the People team and reports directly to the CEO. The board meets four times a year, in March, June, September and December.""",
"""Working hours and leave policy

The standard working week runs from Monday to Friday, with core hours between 10:00 and 16:00. Employees are entitled to 24 days of paid annual leave each calendar year. Up to 8 unused days can be carried forward into the next year; any days beyond that are lost on 31 March.

Sick leave is capped at 12 days per year and a medical certificate is required after 3 consecutive days of absence. New parents receive 26 weeks of paid parental leave, and the partner receives 6 weeks.

Remote work
Employees may work remotely up to 3 days per week after completing their 90-day probation period. Remote staff must be reachable on Slack during core hours. Home-office equipment up to 500 USD is reimbursed once every two years.""",
"""Expenses and travel

Travel must be approved by the line manager at least 7 days in advance. Economy class is mandatory for flights shorter than 6 hours. The daily meal allowance is 45 USD in India and 70 USD abroad. Receipts must be submitted through the Expense Portal within 30 days of the trip, otherwise the claim is rejected.

Laptops are replaced every 3 years. A lost or stolen device must be reported to the IT helpdesk within 24 hours, because the security team has to wipe it remotely.""",
"""Products

Northwind sells two products. Forecaster is the flagship demand-planning tool and costs 1,200 USD per user per year. Shelfwatch is an inventory alert tool priced at 450 USD per store per month. Forecaster version 4.2 was released in October 2025 and added automatic holiday detection.

Support
Customers can reach support by email at any time. Phone support is available from 9:00 to 18:00 IST on weekdays. The target first-response time is 4 business hours for standard tickets and 1 hour for critical outages. Customer satisfaction reached 92 percent in 2025, up from 88 percent in 2024.""",
"""Security and compliance

All employees must complete security awareness training within 14 days of joining and then once every year. Passwords must be at least 14 characters long and are rotated only after a suspected breach. Multi-factor authentication is required for every internal system.

Northwind holds ISO 27001 certification, first obtained in 2021 and renewed in 2024. Customer data is stored in the Mumbai region and backed up nightly to a second region in Hyderabad. Backups are retained for 35 days.""",
]

# (question, key fact expected (lowercase) in answer, expected page; None => unanswerable)
QA = [
    ("Who founded Northwind Analytics?", "meera iyer", 1),
    ("In which year was the company founded?", "2014", 1),
    ("Where is the headquarters located?", "bengaluru", 1),
    ("Who is the CTO?", "daniel brooks", 1),
    ("Who does Priya Nair report to?", "ceo", 1),
    ("How often does the board meet?", "four times", 1),
    ("How many days of annual leave do employees get?", "24", 2),
    ("How many unused leave days can I carry over?", "8", 2),
    ("When do carried-forward leave days expire?", "31 march", 2),
    ("When do I need a medical certificate?", "3 consecutive", 2),
    ("How long is parental leave for new parents?", "26 weeks", 2),
    ("How many days a week can I work from home?", "3 days", 2),
    ("What is the home office equipment reimbursement limit?", "500", 2),
    ("How far in advance must travel be approved?", "7 days", 3),
    ("What is the meal allowance abroad?", "70", 3),
    ("Within how many days must expense receipts be submitted?", "30 days", 3),
    ("What should I do if my laptop is stolen?", "24 hours", 3),
    ("Why must lost devices be reported quickly?", "wipe", 3),
    ("How much does Forecaster cost?", "1,200", 4),
    ("What did version 4.2 add?", "holiday", 4),
    ("What is the first response time for critical outages?", "1 hour", 4),
    ("What was customer satisfaction in 2025?", "92", 4),
    ("How long must passwords be?", "14 characters", 5),
    ("When was ISO 27001 certification renewed?", "2024", 5),
    ("Where are backups of customer data kept?", "hyderabad", 5),
    ("How long are backups retained?", "35 days", 5),
    # unanswerable look-alikes: same vocabulary, fact absent
    ("What is the CEO's salary?", None, None),
    ("How many employees does Northwind have?", None, None),
    ("What is the dress code policy?", None, None),
    ("Who is the Chief Financial Officer?", None, None),
]


def main(use_hf: bool, verbose: bool) -> None:
    docs = [Document(t, "Northwind_Handbook.pdf", i + 1, "pdf") for i, t in enumerate(PAGES)]
    store = VectorStore(get_embedder("hf" if use_hf else "tfidf")).build(build_knowledge_base(docs).chunks)
    ok = cite = refused = 0
    ans_q = [q for q in QA if q[2]]
    un_q = [q for q in QA if not q[2]]
    for q, fact, page in QA:
        r = answer_question(q, store, None)
        if page:
            hit = fact in r["answer"].lower()
            c = any(s["page"] == page for s in r["sources"])
            ok += hit
            cite += c
            if verbose or not hit:
                print(f"{'OK ' if hit else 'MISS'} | {q}\n      -> {r['answer'][:230]!r} | pages={[s['page'] for s in r['sources']]}")
        else:
            good = r["mode"] == "not_found"
            refused += good
            if verbose or not good:
                print(f"{'OK ' if good else 'WRONG(answered)'} | {q}\n      -> {r['answer'][:160]!r}")
    print(f"\nAnswer has key fact : {ok}/{len(ans_q)} = {ok/len(ans_q):.0%}")
    print(f"Correct page cited  : {cite}/{len(ans_q)} = {cite/len(ans_q):.0%}")
    print(f"Correct refusals    : {refused}/{len(un_q)} = {refused/len(un_q):.0%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    main(a.hf, a.verbose)
