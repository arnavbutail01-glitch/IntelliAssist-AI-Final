"""Test doubles."""


class FakeLLM:
    name = "FakeLLM"

    def __init__(self, reply="This is a generated answer [1]."):
        self.reply, self.calls = reply, []

    def generate(self, prompt, system=None):
        self.calls.append((prompt, system))
        return self.reply


class BrokenLLM:
    name = "BrokenLLM"

    def generate(self, prompt, system=None):
        from core.llm import LLMError
        raise LLMError("quota exceeded")
