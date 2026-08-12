#!/usr/bin/env python3
"""One-off resumable controller for the 16-run multi-tenet AFT experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml


EXPERIMENT_DIR = Path(__file__).resolve().parent
MSM_ROOT = EXPERIMENT_DIR.parents[1]
VALUEGEN_ROOT = MSM_ROOT.parents[1]
sys.path.insert(0, str(VALUEGEN_ROOT / "src"))

from valuegen.config import ClusterConfig, load_cluster  # noqa: E402
from valuegen.ground_truth import training  # noqa: E402
from valuegen.ground_truth.evaluation import (  # noqa: E402
    EvalSpec,
    eval_tasks,
    serve_stage,
    wait_ready_sh,
)
from valuegen.slurm import Orchestrator, Stage, Task  # noqa: E402

from prepare import load_combos, write_specs  # noqa: E402


@dataclass(frozen=True)
class Layout:
    artifact: Path
    specs: Path
    staging: Path
    datasets: Path
    evals: Path
    train_root: Path
    merged_root: Path
    gen_hostfile: Path
    judge_hostfile: Path
    train_config: Path


def _load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected a mapping in {path}")
    return payload


def _resolve_path(path: str | Path, base: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else base / candidate


def _git_revision(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _msm_base_revision() -> str:
    return subprocess.run(
        ["git", "merge-base", "HEAD", "valuegen-integration"],
        cwd=MSM_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _experiment_source_digest() -> str:
    digest = hashlib.sha256()
    for name in (
        "README.md",
        "combos.json",
        "controller.py",
        "controller.sbatch",
        "experiment.yaml",
        "prepare.py",
    ):
        path = EXPERIMENT_DIR / name
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _layout(cfg: dict, cluster: ClusterConfig) -> Layout:
    name = cfg["name"]
    artifact = cluster.data / "experiments" / name
    return Layout(
        artifact=artifact,
        specs=artifact / "specs",
        staging=artifact / "msm",
        datasets=artifact / "datasets",
        evals=artifact / "model_evals",
        train_root=cluster.finetune_root / "experiments" / name,
        merged_root=cluster.finetune_root / "merged" / name,
        gen_hostfile=artifact / "slurm" / "generator_host.txt",
        judge_hostfile=artifact / "slurm" / "judge_host.txt",
        train_config=artifact / "train_config.yaml",
    )


def _experiment_inputs(config_path: Path, cfg: dict) -> tuple[list[dict], dict[str, str], Path]:
    combos_path = _resolve_path(cfg["combos"], config_path.parent)
    value_set_path = VALUEGEN_ROOT / "value_sets" / f"{cfg['value_set']}.json"
    values = json.loads(value_set_path.read_text())
    combos = load_combos(combos_path, values)
    return combos, values, combos_path


def _claim(
    config_path: Path,
    cfg: dict,
    combos: list[dict],
    values: dict[str, str],
    layout: Layout,
) -> None:
    layout.artifact.mkdir(parents=True, exist_ok=True)
    layout.staging.mkdir(parents=True, exist_ok=True)
    layout.datasets.mkdir(parents=True, exist_ok=True)
    layout.evals.mkdir(parents=True, exist_ok=True)
    layout.gen_hostfile.parent.mkdir(parents=True, exist_ok=True)
    resolved = {
        "config": cfg,
        "combos": combos,
        "values": values,
        "experiment_source_sha256": _experiment_source_digest(),
        "model_spec_midtraining_base_commit": _msm_base_revision(),
        "valuegen_commit": _git_revision(VALUEGEN_ROOT),
    }
    encoded = json.dumps(resolved, sort_keys=True, separators=(",", ":"))
    record = {
        "experiment_id": f"{cfg['name']}-{hashlib.sha256(encoded.encode()).hexdigest()[:12]}",
        **resolved,
    }
    record_path = layout.artifact / "resolved_experiment.json"
    if record_path.exists():
        existing = json.loads(record_path.read_text())
        if existing != record:
            raise RuntimeError(
                f"{record_path} belongs to different inputs; choose a new experiment name"
            )
    else:
        record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")

    write_specs(combos, values, layout.specs)
    training_cfg = dict(cfg["training"])
    for scheduling_key in ("model", "tag", "time", "mem", "wandb_project"):
        training_cfg.pop(scheduling_key, None)
    if layout.train_config.exists():
        if yaml.safe_load(layout.train_config.read_text()) != training_cfg:
            raise RuntimeError(f"refusing to overwrite mismatched {layout.train_config}")
    else:
        layout.train_config.write_text(yaml.safe_dump(training_cfg, sort_keys=False))


def _q(value: str | Path) -> str:
    return shlex.quote(str(value))


def _generation_source(layout: Layout, cid: str) -> Path:
    return layout.staging / "data" / "ft" / f"{cid}_cot_stripped" / "dataset.jsonl"


def _dataset(layout: Layout, cid: str) -> Path:
    return layout.datasets / cid / "dataset.jsonl"


def _train_output(layout: Layout, cid: str) -> Path:
    return layout.train_root / cid


def _merged(layout: Layout, training_cfg: dict, cid: str) -> Path:
    return layout.merged_root / f"{training_cfg['tag']}_{cid}_merged"


def _generation_stages(
    cfg: dict, combos: list[dict], cluster: ClusterConfig, layout: Layout
) -> list[Stage]:
    pending = [combo for combo in combos if not _dataset(layout, combo["id"]).is_file()]
    if not pending:
        return []
    gen = cfg["generation"]
    layout.gen_hostfile.parent.mkdir(parents=True, exist_ok=True)
    server = serve_stage(
        name="serve_generator",
        model=gen["model"],
        hostfile=layout.gen_hostfile,
        gpus=int(gen["gpus"]),
        mem=str(gen["mem"]),
        time=str(gen["server_time"]),
        env="eval_api",
        load_path=cluster.model_path(gen["model"]),
    )
    prelude = wait_ready_sh(layout.gen_hostfile, "generator")
    tasks: list[Task] = []
    for combo in pending:
        cid = combo["id"]
        source = _generation_source(layout, cid)
        destination = _dataset(layout, cid)
        command = f"""{prelude}
export PYTHONPATH={_q(MSM_ROOT)}
export RUNPOD_API_KEY="${{RUNPOD_API_KEY:-api}}"
cd {_q(layout.staging)}
python -m src.aft.generate_chat \\
    --dataset_name {_q(cid)} \\
    --spec_name {_q(layout.specs / f'{cid}.txt')} \\
    --n_samples {int(gen['n_samples'])} \\
    --questions_per_domain {int(gen['questions_per_domain'])} \\
    --model_name Assistant \\
    --provider_name 'its developer' \\
    --response_style {_q(gen['response_style'])} \\
    --prompt_version {_q(gen['prompt_version'])} \\
    --model_id {_q(gen['model'])} \\
    --max_tokens {int(gen['max_tokens'])} \\
    --disable_thinking {str(bool(gen['disable_thinking'])).lower()} \\
    --max_concurrent_requests {int(gen['max_concurrent_requests'])} \\
    --dedup_threshold {float(gen['dedup_threshold'])} \\
    --use_llm_filter {str(bool(gen['use_llm_filter'])).lower()} \\
    --skip_existing true \\
    --use_vllm_if_model_not_found true \\
    --vllm_base_url "http://$(cat {_q(layout.gen_hostfile)})/v1"
python {_q(EXPERIMENT_DIR / 'prepare.py')} convert \\
    --source {_q(source)} \\
    --output {_q(destination)}
"""
        tasks.append(Task(key=cid, command=command, done=destination))
    generate = Stage(
        name="generate",
        tasks=tasks,
        time=str(gen["client_time"]),
        mem=str(gen["client_mem"]),
        env="msm",
        throttle=len(tasks),
        cpu_partition=True,
        needs_servers=("serve_generator",),
        cancel_servers=("serve_generator",),
    )
    return [server, generate]


def _training_stage(
    cfg: dict, combos: list[dict], cluster: ClusterConfig, layout: Layout
) -> Stage:
    train = cfg["training"]
    model = train["model"]
    tasks: list[Task] = []
    for combo in combos:
        cid = combo["id"]
        output = _train_output(layout, cid)
        merged = _merged(layout, train, cid)
        command = (
            training.train_command(
                algo="sft",
                train_config=layout.train_config,
                dataset=_dataset(layout, cid),
                model=model,
                output_dir=output,
                run_name=f"{cfg['name']}_{train['tag']}_{cid}",
            )
            + "\n"
            + training.merge_command(model, output, merged)
        )
        tasks.append(Task(key=cid, command=command, done=merged / "config.json"))
    return Stage(
        name="train_olmo7b_base",
        tasks=tasks,
        time=str(train["time"]),
        mem=str(train["mem"]),
        gpus=1,
        env="default",
        extra_exports={"WANDB_PROJECT": str(train["wandb_project"])},
    )


def _evaluation_stages(
    cfg: dict, combos: list[dict], cluster: ClusterConfig, layout: Layout
) -> list[Stage]:
    ev = cfg["evaluation"]
    train = cfg["training"]
    layout.evals.mkdir(parents=True, exist_ok=True)
    layout.judge_hostfile.parent.mkdir(parents=True, exist_ok=True)
    spec = EvalSpec(
        scenarios_dir=_resolve_path(ev["scenarios"], VALUEGEN_ROOT),
        output_dir=layout.evals,
        interactive=bool(ev["interactive"]),
        cache=bool(ev["cache"]),
        filter=bool(ev["filter"]),
        user_model=ev["judge"],
        judge_model=ev["judge"],
        user_api_base=layout.judge_hostfile,
        judge_api_base=layout.judge_hostfile,
        wait_for={"judge": layout.judge_hostfile},
    )
    interventions = {
        f"{train['tag']}_{combo['id']}": {"model": str(_merged(layout, train, combo["id"]))}
        for combo in combos
    }
    tasks = eval_tasks(
        spec,
        base_model=train["model"],
        interventions=interventions,
        base_tag=f"{train['tag']}_base",
    )
    if not any(not task.is_done() for task in tasks):
        return []
    judge = serve_stage(
        name="serve_judge",
        model=ev["judge"],
        hostfile=layout.judge_hostfile,
        gpus=int(ev["judge_gpus"]),
        mem=str(ev["judge_mem"]),
        time=str(ev["judge_time"]),
        env="eval_api",
        load_path=cluster.model_path(ev["judge"]),
    )
    evaluate = Stage(
        name="eval_olmo7b_base",
        tasks=tasks,
        time=str(ev["task_time"]),
        mem=str(ev["task_mem"]),
        gpus=1,
        env="default",
        gpu_headroom=int(ev["judge_gpus"]),
        needs_servers=("serve_judge",),
        extra_exports={"OPENAI_API_KEY": "${OPENAI_API_KEY:-dummy}"},
    )
    return [judge, evaluate]


def build_pipeline(config_path: Path, cluster_path: str | None, dry_run: bool) -> Orchestrator:
    cfg = _load_yaml(config_path)
    cluster = load_cluster(cluster_path)
    combos, values, _combos_path = _experiment_inputs(config_path, cfg)
    layout = _layout(cfg, cluster)
    _claim(config_path, cfg, combos, values, layout)
    stages = [
        *_generation_stages(cfg, combos, cluster, layout),
        _training_stage(cfg, combos, cluster, layout),
        *_evaluation_stages(cfg, combos, cluster, layout),
    ]
    return Orchestrator(cfg["name"], stages, cluster, dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verb", choices=("run", "status", "dry-run"))
    parser.add_argument("--config", type=Path, default=EXPERIMENT_DIR / "experiment.yaml")
    parser.add_argument("--cluster", default=None)
    args = parser.parse_args()
    pipeline = build_pipeline(args.config.resolve(), args.cluster, args.verb == "dry-run")
    if args.verb == "status":
        pipeline.status()
    elif args.verb == "dry-run":
        if not pipeline.run():
            raise SystemExit(1)
    elif not pipeline.run():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
