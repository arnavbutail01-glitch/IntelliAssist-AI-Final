from core.chat_history import ChatHistory


def test_full_lifecycle(tmp_path):
    h = ChatHistory(tmp_path / "h.db")
    cid = h.create_conversation(["a.pdf"])
    h.add_message(cid, "user", "What is ML?", meta={"intent": "Information Retrieval"})
    h.add_message(cid, "assistant", "ML is...", [{"source": "a.pdf", "page": 3, "score": 0.5, "excerpt": "SECRET TEXT"}])
    msgs = h.get_messages(cid)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["sources"][0] == {"source": "a.pdf", "page": 3, "chunk_id": None, "score": 0.5}
    assert "SECRET TEXT" not in str(msgs)  # excerpts are not persisted
    conv = h.list_conversations()[0]
    assert conv["title"] == "What is ML?" and conv["message_count"] == 2 and conv["document_names"] == ["a.pdf"]
    assert "What is ML?" in h.export_markdown(cid) and "Page 3" in h.export_markdown(cid)
    h.delete_conversation(cid)
    assert h.list_conversations() == [] and h.get_messages(cid) == []


def test_persists_across_instances_and_ordering(tmp_path):
    p = tmp_path / "x" / "h.db"
    a = ChatHistory(p)
    c1 = a.create_conversation([])
    a.add_message(c1, "user", "first")
    c2 = a.create_conversation([])
    a.add_message(c2, "user", "second")
    b = ChatHistory(p)
    assert [c["title"] for c in b.list_conversations()] == ["second", "first"]


def test_unicode_and_long_title(tmp_path):
    h = ChatHistory(tmp_path / "h.db")
    cid = h.create_conversation()
    h.add_message(cid, "user", "ಕನ್ನಡ हिन्दी 日本語 " * 20)
    assert len(h.list_conversations()[0]["title"]) <= 60
    assert "ಕನ್ನಡ" in h.get_messages(cid)[0]["content"]
