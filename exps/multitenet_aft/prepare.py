#!/usr/bin/env python3
"""Render co-equal specs and convert MSM chat output for valuegen SFT."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


SPEC_HEADER = """The following principles are co-equal parts of the assistant's highest-priority specification.

The assistant is deeply and unconditionally committed to all of these principles. Their order does not indicate priority. The assistant satisfies all of them whenever possible. When they genuinely conflict, the assistant makes a context-sensitive tradeoff that respects each principle rather than permanently subordinating any one of them.

The principles are:
"""


def load_combos(path: Path, values: dict[str, str]) -> list[dict]:
    payload = json.loads(path.read_text())
    combos = payload.get("combos", [])
    if len(combos) != 16:
        raise ValueError(f"expected 16 combinations, found {len(combos)}")
    seen: set[str] = set()
    counts: dict[int, int] = {}
    for combo in combos:
        cid = combo.get("id")
        tenets = combo.get("tenets")
        if not isinstance(cid, str) or not re.fullmatch(r"n[2468]_[0-3]", cid):
            raise ValueError(f"invalid combination id: {cid!r}")
        if cid in seen:
            raise ValueError(f"duplicate combination id: {cid}")
        seen.add(cid)
        if not isinstance(tenets, list) or len(tenets) != int(cid[1]):
            raise ValueError(f"{cid}: membership does not match its declared size")
        if len(set(tenets)) != len(tenets):
            raise ValueError(f"{cid}: duplicate tenet")
        unknown = [tenet for tenet in tenets if tenet not in values]
        if unknown:
            raise ValueError(f"{cid}: unknown tenets {unknown}")
        counts[len(tenets)] = counts.get(len(tenets), 0) + 1
    if counts != {2: 4, 4: 4, 6: 4, 8: 4}:
        raise ValueError(f"expected four combinations per size, found {counts}")
    return combos


def render_spec(tenets: list[str], values: dict[str, str]) -> str:
    registry_order = {name: i for i, name in enumerate(values)}
    ordered = sorted(tenets, key=registry_order.__getitem__)
    bullets = "\n".join(f"- {values[tenet].strip()}" for tenet in ordered)
    return SPEC_HEADER + bullets + "\n"


def write_specs(combos: list[dict], values: dict[str, str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for combo in combos:
        path = output_dir / f"{combo['id']}.txt"
        expected = render_spec(combo["tenets"], values)
        if path.exists() and path.read_text() != expected:
            raise RuntimeError(f"refusing to overwrite mismatched spec: {path}")
        if not path.exists():
            path.write_text(expected)


def convert_dataset(source: Path, output: Path) -> int:
    if not source.is_file():
        raise FileNotFoundError(source)
    rows: list[dict] = []
    with source.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            messages = json.loads(line)["messages"]
            if [message.get("role") for message in messages] != ["user", "assistant"]:
                raise ValueError(f"{source}:{line_number}: expected user/assistant messages")
            if "<think>" in messages[1].get("content", ""):
                raise ValueError(f"{source}:{line_number}: unstripped <think> block")
            rows.append({"prompt": [messages[0]], "chosen": [messages[1]]})
    if not rows:
        raise ValueError(f"no usable examples in {source}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, output)
    print(f"converted {len(rows)} examples: {source} -> {output}")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    specs = subparsers.add_parser("specs")
    specs.add_argument("--combos", type=Path, required=True)
    specs.add_argument("--values", type=Path, required=True)
    specs.add_argument("--output-dir", type=Path, required=True)

    convert = subparsers.add_parser("convert")
    convert.add_argument("--source", type=Path, required=True)
    convert.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "convert":
        convert_dataset(args.source, args.output)
        return

    values = json.loads(args.values.read_text())
    combos = load_combos(args.combos, values)
    write_specs(combos, values, args.output_dir)
    print(f"rendered {len(combos)} specs in {args.output_dir}")


if __name__ == "__main__":
    main()
