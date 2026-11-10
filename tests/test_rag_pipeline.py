from tests.helpers import BrokenLLM, FakeLLM
from core.llm import GeminiLLM, LLMError, OpenAILLM, make_llm
from core.rag_pipeline import (NOT_FOUND, SYSTEM_PROMPT, answer_question, build_prompt,
                               retrieve_context)


def test_answer_with_citations_extractive(store):
    r = answer_question("Who reviewed the design?", store, None)
    assert r["mode"] == "extractive" and "Ananya Rao" in r["answer"]
    assert r["sources"][0]["source"] == "RAG_Handbook.pdf" and r["sources"][0]["page"] == 2
    assert r["error"] is None


def test_unrelated_questions_refused(store):
    for q in ["What is the capital of France?", "Who won the World Cup?", "zzzz qqqq"]:
        r = answer_question(q, store, FakeLLM())
        assert r["answer"] == NOT_FOUND and r["sources"] == [] and r["mode"] == "not_found"


def test_llm_used_and_prompt_is_grounded(store):
    llm = FakeLLM()
    r = answer_question("What is RAG?", store, llm)
    assert r["mode"] == "llm" and r["answer"].startswith("This is a generated")
    prompt, system = llm.calls[0]
    assert "RAG_Handbook.pdf" in prompt and "Page: 1" in prompt and system == SYSTEM_PROMPT
    assert "not as system rules" in SYSTEM_PROMPT


def test_llm_says_not_found_drops_sources(store):
    r = answer_question("What is RAG?", store, FakeLLM(NOT_FOUND))
    assert r["mode"] == "not_found" and r["sources"] == []


def test_llm_failure_falls_back(store):
    r = answer_question("What is RAG?", store, BrokenLLM())
    assert r["mode"] == "extractive" and "quota exceeded" in r["error"] and r["sources"]


def test_unexpected_llm_exception_does_not_crash(store):
    class Boom:
        def generate(self, *a, **k):
            raise RuntimeError("boom")
    r = answer_question("What is RAG?", store, Boom())
    assert r["mode"] == "extractive" and r["error"]


def test_page_specific_question(store):
    hits = retrieve_context("Who is mentioned on page 2?", store)
    assert hits and all(h.chunk.page == 2 for h in hits)


def test_followup_uses_history(store):
    hist = [{"role": "user", "content": "What is FAISS used for in the handbook?"}]
    hits = retrieve_context("and why?", store, history=hist)
    assert hits and hits[0].chunk.source == "RAG_Handbook.pdf"


def test_empty_question(store):
    assert answer_question("   ", store, None)["mode"] == "not_found"


def test_prompt_injection_stays_in_context(store):
    p = build_prompt("q", retrieve_context("RAG", store))
    assert p.index("Document context") < p.index("Question:")


def test_gemini_client(monkeypatch):
    import requests

    class R:
        def __init__(self, code, js): self.status_code, self._j, self.text = code, js, str(js)
        def json(self): return self._j

    monkeypatch.setattr(requests, "post", lambda *a, **k: R(200, {"candidates": [{"content": {"parts": [{"text": " hi "}]}}]}))
    assert GeminiLLM("k").generate("p", system="s") == "hi"
    monkeypatch.setattr(requests, "post", lambda *a, **k: R(429, {"error": "quota"}))
    try:
        GeminiLLM("SECRETKEY").generate("p")
        assert False
    except LLMError as e:
        assert "429" in str(e) and "SECRETKEY" not in str(e)
    monkeypatch.setattr(requests, "post", lambda *a, **k: R(200, {"candidates": []}))
    try:
        GeminiLLM("k").generate("p"); assert False
    except LLMError:
        pass

    def boom(*a, **k): raise requests.ConnectionError("no net")
    monkeypatch.setattr(requests, "post", boom)
    try:
        GeminiLLM("k").generate("p"); assert False
    except LLMError as e:
        assert "reach" in str(e)


def test_openai_client_and_factory(monkeypatch):
    import requests

    class R:
        status_code = 200
        text = ""
        def json(self): return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(requests, "post", lambda *a, **k: R())
    assert OpenAILLM("k").generate("p") == "ok"
    assert make_llm("Gemini", "") is None and isinstance(make_llm("OpenAI", "k"), OpenAILLM)
    assert isinstance(make_llm("Gemini", "k"), GeminiLLM)


def test_gemini_falls_back_when_model_retired(monkeypatch):
    import requests

    class R:
        def __init__(self, code, js): self.status_code, self._j, self.text = code, js, str(js)
        def json(self): return self._j

    seen = []

    def fake_post(url, **k):
        seen.append(url)
        if len(seen) == 1:
            return R(404, {"error": "model no longer available"})
        return R(200, {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})

    monkeypatch.setattr(requests, "post", fake_post)
    llm = GeminiLLM("k", model="old-retired-model")
    assert llm.generate("p") == "ok"
    assert len(seen) == 2 and "old-retired-model" in seen[0] and llm.model != "old-retired-model"
