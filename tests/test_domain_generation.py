import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.aft.generate_chat import SpecAlignedChatGenerator


class _StubDomainGenerator(SpecAlignedChatGenerator):
    def __init__(self, config, responses):
        self.config = config
        self.domain_template = "ORIGINAL count={count} spec={spec}{existing_domains}"
        self.domains = []
        self._responses = iter(responses)
        self.seen_prompts = []

    async def _api_call(self, prompt, **kwargs):
        self.seen_prompts.append(prompt.messages[0].content)
        return next(self._responses)


def _config(source_dir: Path, spec: str = "Always give careful, useful answers."):
    return SimpleNamespace(
        source_dir=source_dir,
        domains_file=None,
        skip_existing=True,
        n_samples=5,
        questions_per_domain=1,
        domain_batch_size=50,
        spec_content=spec,
    )


def _tagged_domains(count: int = 5) -> str:
    items = "\n".join(
        f"{index}. Diagnostic conversation domain number {index}"
        for index in range(1, count + 1)
    )
    return f"<output>\n{items}\n</output>"


@pytest.mark.parametrize("spec", ["Spec for value alpha.", "Spec for value beta."])
def test_domain_generation_uses_uniform_output_only_recovery(tmp_path, spec):
    generator = _StubDomainGenerator(
        _config(tmp_path, spec),
        [
            "1. Raw numbered reasoning must not be accepted as final output.",
            _tagged_domains(),
        ],
    )

    domains = asyncio.run(generator.generate_domains())

    assert len(domains) == 5
    assert generator.seen_prompts[0] == f"ORIGINAL count=5 spec={spec}"
    assert "Return only a numbered list inside one complete <output> block" in generator.seen_prompts[1]
    assert "Do not include\nanalysis, planning" in generator.seen_prompts[1]
    assert spec in generator.seen_prompts[1]


def test_domain_generation_fails_explicitly_after_three_unparseable_attempts(tmp_path):
    generator = _StubDomainGenerator(
        _config(tmp_path),
        ["raw numbered list only"] * 3,
    )

    with pytest.raises(RuntimeError, match="no parseable domains after 3 attempts"):
        asyncio.run(generator.generate_domains())

    assert len(generator.seen_prompts) == 3
    assert generator.seen_prompts[0].startswith("ORIGINAL")
    assert all("Return only a numbered list" in prompt for prompt in generator.seen_prompts[1:])
    assert not (tmp_path / "domains.jsonl").exists()
