"""Generate sample documents (PDF, DOCX, TXT) in data/sample_documents/.
Needs: reportlab, python-docx.   Run: python scripts/make_samples.py"""
from pathlib import Path

import docx
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parent.parent / "data" / "sample_documents"
OUT.mkdir(parents=True, exist_ok=True)

AI_NOTES = """Chapter 1: Introduction to Artificial Intelligence

Artificial intelligence (AI) is the field of computer science that builds systems able to perform tasks that normally require human intelligence. These tasks include learning from experience, understanding language, recognising images and making decisions. AI is transforming education, healthcare, finance and transportation.

Machine learning is a branch of AI in which algorithms learn patterns from data instead of following hand-written rules. Supervised learning uses labelled examples, unsupervised learning finds hidden structure in unlabeled data, and reinforcement learning learns from rewards and penalties.

Chapter 2: Applications of Machine Learning

Machine learning is used for predictive analytics, image recognition, recommendation systems and fraud detection. Hospitals use it to detect diseases from medical scans. Online shops use recommendation systems to suggest products. Banks use anomaly detection to flag suspicious transactions.

Chapter 3: Ethics and Limitations

AI systems can inherit bias from their training data and may produce unfair results. Transparency, privacy protection and human oversight are essential. The author concludes that AI should augment human decision making rather than replace it, and that responsible governance is the key to trustworthy AI.
"""

NLP_SECTIONS = [
    ("Natural Language Processing Notes", [
        "Natural language processing (NLP) enables computers to understand, interpret and generate human language. Typical NLP tasks include sentiment analysis, named entity recognition, machine translation and question answering.",
        "Tokenization splits text into words or sub-words. TF-IDF weights a term by how often it appears in a document and how rare it is across the collection, which makes it a classic technique for keyword search.",
        "Embeddings are dense numerical vectors that represent the meaning of text. Sentences with similar meaning have vectors that are close together, so embeddings power semantic search. BERT-style models such as Sentence-BERT produce high quality sentence embeddings.",
    ]),
    ("Evaluating Language Models", [
        "Precision measures how many retrieved items are relevant, while recall measures how many relevant items were retrieved. For retrieval systems, Precision at K is a common metric. Latency measures how quickly a system answers.",
    ]),
]

REVIEW = """Customer feedback on the IntelliAssist pilot

Overall the pilot users were very happy. The search is fast and the answers are accurate and helpful. Several students said the interface is intuitive and they would recommend it to classmates.

However, the first version had problems. The upload screen was slow and sometimes crashed on large files. One reviewer called the export feature confusing and the error messages poor.

The deployment team reported that the server uses 2 GB of memory. The index is rebuilt each time the application starts.
"""

RAG_PAGES = [
    ("Retrieval-Augmented Generation Handbook", "Page one introduces the handbook. Retrieval-Augmented Generation, or RAG, combines a retrieval system with a language model. Instead of relying only on what the model memorised during training, a RAG system first looks up relevant passages in a private knowledge base and then asks the model to answer using those passages. This reduces hallucination and allows answers to be traced to their sources."),
    ("Embeddings and Vector Databases", "Page two explains how retrieval works. Each document chunk is converted into an embedding vector by an embedding model. The vectors are stored in a vector database such as FAISS, which can quickly find the nearest vectors to a query. Typical chunk sizes are between 500 and 1000 characters with a small overlap so that context is not lost at chunk boundaries. Dr. Ananya Rao from Bengaluru reviewed this design."),
    ("Conclusion", "Page three concludes the handbook. The authors conclude that RAG is the most practical way to build trustworthy question answering over private documents. They recommend evaluating retrieval precision, answer correctness and citation accuracy, and they advise always showing source citations to the user."),
]


def make_pdf():
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(OUT / "RAG_Handbook.pdf"), pagesize=A4)
    story = []
    for i, (title, body) in enumerate(RAG_PAGES):
        story += [Paragraph(title, styles["Heading1"]), Spacer(1, 12), Paragraph(body, styles["BodyText"])]
        if i < len(RAG_PAGES) - 1:
            story.append(PageBreak())
    doc.build(story)


def make_docx():
    d = docx.Document()
    for heading, paras in NLP_SECTIONS:
        d.add_heading(heading, level=1)
        for p in paras:
            d.add_paragraph(p)
    t = d.add_table(rows=3, cols=2)
    for r, (a, b) in enumerate([("Metric", "Meaning"), ("Precision@K", "Share of top-K results that are relevant"),
                                ("Recall@K", "Share of relevant items found in the top K")]):
        t.cell(r, 0).text, t.cell(r, 1).text = a, b
    d.save(str(OUT / "NLP_Notes.docx"))


if __name__ == "__main__":
    (OUT / "AI_Notes.txt").write_text(AI_NOTES, encoding="utf-8")
    (OUT / "Pilot_Feedback.txt").write_text(REVIEW, encoding="utf-8")
    make_pdf()
    make_docx()
    print("Samples written to", OUT)
