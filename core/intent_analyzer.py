"""Rule-based intent classification of user queries."""
from __future__ import annotations

import re
from typing import Dict

# (intent, category, regex, explanation) - first match wins
RULES = [
    ("Summarization", "Summarisation", r"\b(summari[sz]e|summary|tl;?dr|overview|gist|recap|brief me|in short)\b",
     "The user wants a condensed version of the content."),
    ("Comparison", "Analysis", r"\b(compare|comparison|differences?|differ|versus|vs\.?|similarities|contrast)\b",
     "The user wants two or more things compared."),
    ("Extraction", "Extraction", r"\b(list|key points?|extract|enumerate|bullet|main points?|highlights?|takeaways?|keywords?)\b",
     "The user wants specific items pulled out of the document."),
    ("Opinion/Analysis", "Analysis", r"\b(opinions?|stance|viewpoint|perspective|believe[sd]?|argue[sd]?|evaluate|critique|pros and cons|think about|assess|tone)\b",
     "The user wants the author's viewpoint or an evaluation."),
    ("Explanation", "Explanation", r"\b(explain|why|how (does|do|did|is|are|can|to)|describe|elaborate|clarify|walk me through|what does .* mean)\b",
     "The user wants a concept explained in detail."),
    ("Information Retrieval", "Question Answering", r"\b(what|who|when|where|which|define|definition|find|locate|how (many|much)|is there|are there|does)\b",
     "The user is looking for a specific fact or definition."),
]
GREETING = re.compile(r"^\s*(hi|hello|hey|good (morning|afternoon|evening)|thanks|thank you)\b[\s!.?]*$", re.I)


def analyze_intent(query: str) -> Dict:
    """Return {intent, category, confidence, explanation}."""
    q = (query or "").strip()
    if not q:
        return {"intent": "Unknown", "category": "Other", "confidence": 0.0, "explanation": "Empty query."}
    if GREETING.match(q):
        return {"intent": "Greeting", "category": "Conversation", "confidence": 0.9,
                "explanation": "A greeting or thanks, not a document question."}
    for intent, category, pattern, why in RULES:
        if re.search(pattern, q, re.I):
            return {"intent": intent, "category": category, "confidence": 0.85, "explanation": why}
    return {"intent": "Information Retrieval", "category": "Question Answering", "confidence": 0.5,
            "explanation": "No specific pattern matched; treated as a general document question."}
