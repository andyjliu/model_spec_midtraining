"""run_backfill: defaults reproduce the historical loop; the flags close the gap."""
import asyncio
import json
import re
from types import SimpleNamespace

import pytest

from src.aft import generate_chat


class StubGenerator:
    """Deterministic stand-in: unlimited fresh questions, fixed judge pass rate."""

    def __init__(self, pass_rate: float, config):
        self.pass_rate = pass_rate
        self.config = config
        self._n = 0
        self.rounds = []

    async def _generate_backfill_questions(self, n, existing_questions):
        self.rounds.append(n)
        out = []
        while len(out) < n:
            self._n += 1
            q = f"What should I do about backfill question {self._n}?"
            if q not in existing_questions:
                out.append({"question": q, "domain": "personal finance advice"})
        return out

    async def _generate_backfill_responses(self, questions):
        return [{"question": q["question"], "response": "r", "domain": q["domain"]} for q in questions]

    async def filter_examples(self, qa_pairs):
        kept, removed = [], []
        for qa in qa_pairs:
            ex = {
                "messages": [
                    {"role": "user", "content": qa["question"]},
                    {"role": "assistant", "content": qa["response"]},
                ],
                "metadata": {"domain": qa["domain"]},
            }
            # Deterministic per question (a real judge is, via its cache):
            # keep a fixed fraction, e.g. pass_rate 0.5 -> every other id.
            idx = int(re.sub(r"\D", "", qa["question"]))
            (kept if self.pass_rate == 1.0
             or (idx * self.pass_rate) % 1 < self.pass_rate - 1e-9
             else removed).append(ex)
        return kept, removed

    def dedup_backfill_questions(self, questions, kept_texts):
        return questions, 0


def _config(tmp_path, **flags):
    cfg = dict(
        n_samples=5000, source_dir=tmp_path, dedup_threshold=0.91,
        backfill_max_rounds=3, backfill_pass_rate_scaled=False,
        backfill_until_full=False, backfill_dedup=False,
        persist_backfill_responses=True,
    )
    cfg.update(flags)
    return SimpleNamespace(**cfg)


def _base(n_kept, n_removed):
    """A base round already filtered: n_kept kept, n_removed removed."""
    qa, kept, removed = [], [], []
    for i in range(n_kept + n_removed):
        q = f"What should I do about base question {i}?"
        qa.append({"question": q, "response": "r", "domain": "personal finance advice"})
        ex = {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": "r"}],
              "metadata": {"domain": "personal finance advice"}}
        (kept if i < n_kept else removed).append(ex)
    return qa, kept, removed


def _log(tmp_path):
    return [json.loads(l) for l in (tmp_path / "backfill_log.jsonl").read_text().splitlines()]


def test_defaults_reproduce_historical_stop_short(tmp_path):
    # no_preachy_tone shape: 2537 kept of 4426 at ~0.5 pass; flat 1.1x rounds
    # close only half the gap each, so three rounds end short of 5000.
    cfg = _config(tmp_path)
    gen = StubGenerator(0.5, cfg)
    qa, kept, removed = _base(2537, 1889)
    rounds = asyncio.run(generate_chat.run_backfill(gen, cfg, qa, kept, removed))
    assert rounds == 3
    assert gen.rounds[0] == int((5000 - 2537) * 1.1)
    assert len(kept) < 5000
    assert _log(tmp_path)[-1]["stop_reason"] == "round_cap"


def test_defaults_stop_inside_slack(tmp_path):
    cfg = _config(tmp_path)
    gen = StubGenerator(1.0, cfg)
    qa, kept, removed = _base(4850, 100)
    rounds = asyncio.run(generate_chat.run_backfill(gen, cfg, qa, kept, removed))
    assert rounds == 0 and len(kept) == 4850
    assert _log(tmp_path)[-1] == {"stop_reason": "target", "rounds": 0, "final": 4850, "gap": 150}


def test_fixed_flags_reach_target(tmp_path):
    cfg = _config(tmp_path, backfill_pass_rate_scaled=True, backfill_until_full=True)
    gen = StubGenerator(0.5, cfg)
    qa, kept, removed = _base(2537, 1889)
    rounds = asyncio.run(generate_chat.run_backfill(gen, cfg, qa, kept, removed))
    assert len(kept) >= 5000
    assert rounds <= 3
    # first round sized by the base pass rate, not a flat 1.1x
    pr = 2537 / (2537 + 1889)
    assert gen.rounds[0] == generate_chat.math.ceil((5000 - 2537) / pr * 1.1)
    log = _log(tmp_path)
    assert log[-1]["stop_reason"] == "target"
    assert [r["round"] for r in log[:-1]] == list(range(1, rounds + 1))
    assert log[0]["pass_rate"] == pytest.approx(pr)


def test_fixed_flags_close_small_gap(tmp_path):
    cfg = _config(tmp_path, backfill_pass_rate_scaled=True, backfill_until_full=True)
    gen = StubGenerator(1.0, cfg)
    qa, kept, removed = _base(4850, 100)
    rounds = asyncio.run(generate_chat.run_backfill(gen, cfg, qa, kept, removed))
    assert rounds == 1 and len(kept) >= 5000


def test_persist_and_resume(tmp_path):
    cfg = _config(tmp_path, backfill_pass_rate_scaled=True, backfill_until_full=True)
    gen = StubGenerator(0.5, cfg)
    qa, kept, removed = _base(2537, 1889)
    asyncio.run(generate_chat.run_backfill(gen, cfg, qa, kept, removed))
    persisted = [json.loads(l) for l in (tmp_path / "backfill_responses.jsonl").read_text().splitlines()]
    assert len(persisted) == len(qa) - 4426
    assert not (tmp_path / "responses.jsonl").exists()  # base record untouched

    # A restarted run re-filters the persisted responses and never regenerates.
    class NoGen(StubGenerator):
        async def _generate_backfill_questions(self, n, existing):
            raise AssertionError("regenerated after persist")

    gen2 = NoGen(0.5, cfg)
    qa2, kept2, removed2 = _base(2537, 1889)
    rounds = asyncio.run(generate_chat.run_backfill(gen2, cfg, qa2, kept2, removed2))
    assert rounds == 0 and len(kept2) >= 5000
    resumed = [r for r in _log(tmp_path) if r.get("round") == 0]
    assert resumed and resumed[-1]["resumed"] == len(persisted)


def test_dedup_backfill_questions_never_drops_pool(monkeypatch):
    def fake_dedup(texts, threshold):
        pairs = []
        for j in range(len(texts)):
            for i in range(j):
                if texts[i] == texts[j]:
                    pairs.append((i, j, 1.0))
                    break
        return {j for _, j, _ in pairs}, pairs

    monkeypatch.setattr(generate_chat, "dedup_by_cosine_similarity", fake_dedup)
    gen = object.__new__(generate_chat.SpecAlignedChatGenerator)
    gen.config = SimpleNamespace(dedup_threshold=0.91)
    kept_texts = ["a", "b"]
    new = [{"question": q, "domain": "personal finance advice"} for q in ["a", "c", "c", "d"]]
    survivors, dropped = gen.dedup_backfill_questions(new, kept_texts)
    assert [q["question"] for q in survivors] == ["c", "d"]
    assert dropped == 2
    assert gen.dedup_backfill_questions([], kept_texts) == ([], 0)


def test_run_backfill_dedup_flag_calls_helper(tmp_path):
    cfg = _config(tmp_path, backfill_dedup=True, backfill_until_full=True,
                  backfill_pass_rate_scaled=True)
    calls = []

    class Gen(StubGenerator):
        def dedup_backfill_questions(self, questions, kept_texts):
            calls.append((len(questions), len(kept_texts)))
            return questions[:-1], 1

    gen = Gen(1.0, cfg)
    qa, kept, removed = _base(4990, 0)
    asyncio.run(generate_chat.run_backfill(gen, cfg, qa, kept, removed))
    assert calls and calls[0][1] == 4990
    assert _log(tmp_path)[0]["dedup_dropped"] == 1
