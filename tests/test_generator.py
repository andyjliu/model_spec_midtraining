import asyncio
from types import SimpleNamespace

from src.aft import generator as generator_module


class _Generator(generator_module.ChatGenerator):
    async def generate_questions(self):
        raise NotImplementedError

    async def generate_responses(self, questions):
        raise NotImplementedError

    async def filter_examples(self, qa_pairs):
        raise NotImplementedError

    def save_final_dataset(self, examples):
        raise NotImplementedError


def _instance(disable_thinking):
    instance = object.__new__(_Generator)
    instance.config = SimpleNamespace(
        disable_thinking=disable_thinking,
        api=object(),
        model_id="Qwen/Qwen3.6-27B",
        max_tokens=2048,
        temperature=1.0,
    )
    instance.semaphore = asyncio.Semaphore(1)
    return instance


def test_api_call_disables_thinking_for_qwen_template(monkeypatch):
    seen = {}

    async def fake_call(**kwargs):
        seen.update(kwargs)
        return "answer"

    monkeypatch.setattr(generator_module, "single_prompt_api_call", fake_call)
    result = asyncio.run(_instance(True)._api_call("prompt"))

    assert result == "answer"
    assert seen["chat_template_kwargs"] == {"enable_thinking": False}


def test_api_call_preserves_default_template_behavior(monkeypatch):
    seen = {}

    async def fake_call(**kwargs):
        seen.update(kwargs)
        return "answer"

    monkeypatch.setattr(generator_module, "single_prompt_api_call", fake_call)
    asyncio.run(_instance(False)._api_call("prompt"))

    assert "chat_template_kwargs" not in seen
