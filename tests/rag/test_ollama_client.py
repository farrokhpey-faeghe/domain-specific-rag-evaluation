import httpx
import ollama
import pytest

from deskrag.core.interfaces import InfraError
from deskrag.rag.llm.ollama_client import OllamaClient

MESSAGES = [{"role": "user", "content": "How much is a transcript?"}]


def reply(content="{}", model="llama3.2:3b", done_reason="stop"):
    return ollama.ChatResponse(
        model=model,
        message=ollama.Message(role="assistant", content=content),
        done=True,
        done_reason=done_reason,
        prompt_eval_count=120,
        eval_count=24,
    )


class StubOllama:
    """Stands in for ollama.Client: records chat calls and plays back scripted results."""

    def __init__(self, results=(), models=(), version="0.40.3"):
        self.results = list(results)  # ChatResponses to return, or exceptions to raise, in order
        self.models = models  # (name, digest) pairs that `ollama list` would show
        self.version = version
        self.chat_calls = []

    def chat(self, **kwargs):
        self.chat_calls.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def list(self):
        return ollama.ListResponse(
            models=[ollama.ListResponse.Model(model=name, digest=d) for name, d in self.models]
        )

    def _request_raw(self, method, path):
        assert (method, path) == ("GET", "/api/version")
        return httpx.Response(200, json={"version": self.version})


def no_sleep(seconds):
    pass


def test_chat_sends_the_schema_and_options_with_thinking_off():
    stub = StubOllama([reply()])
    OllamaClient("llama3.2:3b", client=stub).chat(
        MESSAGES, json_schema={"type": "object"}, options={"seed": 42}
    )
    assert stub.chat_calls == [
        {
            "model": "llama3.2:3b",
            "messages": MESSAGES,
            "format": {"type": "object"},
            "options": {"seed": 42},
            "think": False,
            "stream": False,
        }
    ]


def test_chat_maps_the_ollama_reply_to_an_llm_response():
    stub = StubOllama([reply('{"abstain": true}', done_reason="length")])
    response = OllamaClient("llama3.2:3b", client=stub).chat(MESSAGES)
    assert response.text == '{"abstain": true}'
    assert response.model == "llama3.2:3b"
    assert response.done_reason == "length"
    assert (response.prompt_tokens, response.completion_tokens) == (120, 24)
    assert response.latency_s >= 0


def test_a_reply_from_another_model_is_an_infra_error():
    client = OllamaClient("llama3.2:3b", client=StubOllama([reply(model="gemma3:4b")]))
    with pytest.raises(InfraError, match="gemma3:4b"):
        client.chat(MESSAGES)


def test_an_untagged_model_name_means_latest():
    client = OllamaClient("llama3.2", client=StubOllama([reply(model="llama3.2:latest")]))
    assert client.chat(MESSAGES).model == "llama3.2:latest"


def test_retries_twice_with_backoff():
    sleeps = []
    stub = StubOllama([ConnectionError("down"), httpx.ReadTimeout("slow"), reply()])
    response = OllamaClient("llama3.2:3b", client=stub, sleep=sleeps.append).chat(MESSAGES)
    assert response.done_reason == "stop"
    assert sleeps == [1.0, 2.0]
    assert len(stub.chat_calls) == 3


def test_gives_up_after_three_attempts_with_an_infra_error():
    stub = StubOllama([ollama.ResponseError("runner crashed", 500)] * 3)
    with pytest.raises(InfraError, match="3 attempts"):
        OllamaClient("llama3.2:3b", client=stub, sleep=no_sleep).chat(MESSAGES)
    assert len(stub.chat_calls) == 3


def test_model_digests_by_the_names_asked_for():
    stub = StubOllama(models=[("llama3.2:3b", "a80c"), ("nomic-embed-text:latest", "0a10")])
    client = OllamaClient("llama3.2:3b", client=stub)
    assert client.model_digests(["llama3.2:3b", "nomic-embed-text"]) == {
        "llama3.2:3b": "a80c",
        "nomic-embed-text": "0a10",
    }


def test_a_model_that_is_not_pulled_says_how_to_pull_it():
    client = OllamaClient("gemma3:4b", client=StubOllama(models=[("llama3.2:3b", "a80c")]))
    with pytest.raises(InfraError, match="ollama pull gemma3:4b"):
        client.model_digests(["gemma3:4b"])


def test_server_version():
    client = OllamaClient("llama3.2:3b", client=StubOllama(version="0.40.3"))
    assert client.server_version() == "0.40.3"
