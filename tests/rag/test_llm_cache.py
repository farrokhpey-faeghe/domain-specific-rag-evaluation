from deskrag.core.fakes import FakeLLM
from deskrag.rag.llm.cache import CachedLLM, cache_enabled

MESSAGES = [{"role": "user", "content": "How much is a transcript?"}]
REQUEST = {"messages": MESSAGES, "json_schema": {"type": "object"}, "options": {"seed": 42}}


def test_a_repeated_request_is_answered_from_disk(tmp_path):
    inner = FakeLLM({"transcript": '{"abstain": false, "answer": "$10", "citations": []}'})
    llm = CachedLLM(inner, identity={"digest": "a80c"}, cache_dir=tmp_path)
    first = llm.chat(**REQUEST)
    assert llm.chat(**REQUEST) == first
    assert len(inner.calls) == 1
    assert len(list(tmp_path.glob("*/*.json"))) == 1


def test_messages_schema_options_and_identity_are_all_in_the_key(tmp_path):
    inner = FakeLLM()
    CachedLLM(inner, {"digest": "a"}, tmp_path).chat(**REQUEST)
    variants = [
        ({"digest": "a"}, {**REQUEST, "messages": [{"role": "user", "content": "other"}]}),
        ({"digest": "a"}, {**REQUEST, "json_schema": {"type": "array"}}),
        ({"digest": "a"}, {**REQUEST, "options": {"seed": 7}}),
        ({"digest": "b"}, REQUEST),
    ]
    for identity, request in variants:
        CachedLLM(inner, identity, tmp_path).chat(**request)
    assert len(inner.calls) == 5


def test_the_cache_is_off_unless_asked_for(monkeypatch):
    monkeypatch.delenv("DESKRAG_LLM_CACHE", raising=False)
    assert not cache_enabled()
    monkeypatch.setenv("DESKRAG_LLM_CACHE", "1")
    assert cache_enabled()
