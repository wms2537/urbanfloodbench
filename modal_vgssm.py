"""
Modal launcher for UrbanFloodBench VGSSM training.

Usage:
  modal run modal_vgssm.py --train-args "<train_vgssm_standalone.py args>"
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import threading
from pathlib import Path

import modal


APP_NAME = "urbanfloodbench-vgssm-train"
DATA_VOLUME_NAME = "urbanfloodbench-data"
CKPT_VOLUME_NAME = "urbanfloodbench-checkpoints"
LOGS_VOLUME_NAME = "urbanfloodbench-logs"

REMOTE_SRC = "/root/src"
REMOTE_WORKDIR = "/workspace/urbanfloodbench"
REMOTE_DATA = "/vol/data"
REMOTE_CKPT = "/vol/checkpoints"
REMOTE_LOGS = "/vol/logs"
GPU_MONITOR_INTERVAL_SEC = 30


def _ignore_local_path(path: Path) -> bool:
    rel = path.as_posix().lstrip("./")
    if rel in ("", "."):
        return False
    blocked_prefixes = (
        ".git/",
        "__pycache__/",
        "data/",
        "checkpoints/",
        "logs/",
        "lightning_logs/",
    )
    blocked_suffixes = (
        ".parquet",
        ".ckpt",
        ".log",
        ".csv.gz",
        ".pt",
        ".pth",
    )
    if any(rel.startswith(prefix) for prefix in blocked_prefixes):
        return True
    if any(rel.endswith(suffix) for suffix in blocked_suffixes):
        return True
    return False


def _get_cli_value(args: list[str], flag: str) -> str | None:
    """Read a simple '--flag value' CLI argument from tokenized args."""
    for i, token in enumerate(args):
        if token == flag and i + 1 < len(args):
            return args[i + 1]
    return None


def _set_cli_value(args: list[str], flag: str, value: str) -> None:
    """Set or append a simple '--flag value' CLI argument."""
    for i, token in enumerate(args):
        if token == flag:
            if i + 1 < len(args):
                args[i + 1] = value
            else:
                args.append(value)
            return
    args.extend([flag, value])


def _resolve_resume_checkpoint(train_tokens: list[str]) -> str | None:
    """
    Prefer same-experiment last.ckpt for preemption-safe resume.

    Priority:
    1) checkpoints/model_{model_id}/{exp_name}/last.ckpt (if exists)
    2) user-provided --checkpoint (if exists)
    """
    model_id = _get_cli_value(train_tokens, "--model_id")
    exp_name = _get_cli_value(train_tokens, "--exp_name")
    user_ckpt = _get_cli_value(train_tokens, "--checkpoint")

    if model_id and exp_name:
        exp_last = Path(REMOTE_CKPT) / f"model_{model_id}" / exp_name / "last.ckpt"
        if exp_last.exists():
            return str(exp_last)

    if user_ckpt and Path(user_ckpt).exists():
        return user_ckpt

    return None


def _looks_like_dataset_root(candidate: Path) -> bool:
    required = [
        candidate / "Model_1" / "train" / "1d_nodes_static.csv",
        candidate / "Model_1" / "test" / "1d_nodes_static.csv",
        candidate / "Model_2" / "train" / "1d_nodes_static.csv",
        candidate / "Model_2" / "test" / "1d_nodes_static.csv",
    ]
    return all(path.exists() for path in required)


app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry("pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime")
    .apt_install(
        "git",
        "rsync",
        "build-essential",
        "libgomp1",
        "libgl1",
        "libglib2.0-0",
        "gdal-bin",
        "libgdal-dev",
    )
    .pip_install(
        "pytorch-lightning==2.4.0",
        "pandas==2.2.3",
        "numpy==1.26.4",
        "scikit-learn==1.5.2",
        "networkx==3.3",
        "pyarrow==18.1.0",
        "geopandas==1.0.1",
        "pyogrio==0.10.0",
        "shapely==2.0.6",
        "fiona==1.10.1",
        "tensorboard==2.18.0",
        "tqdm==4.67.1",
    )
    .run_commands(
        "pip install --no-cache-dir "
        "torch-scatter torch-sparse torch-cluster torch-spline-conv "
        "-f https://data.pyg.org/whl/torch-2.2.0+cu121.html",
        "pip install --no-cache-dir torch-geometric==2.6.1",
    )
    .add_local_dir(".", remote_path=REMOTE_SRC, ignore=_ignore_local_path)
)

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
ckpt_volume = modal.Volume.from_name(CKPT_VOLUME_NAME, create_if_missing=True)
logs_volume = modal.Volume.from_name(LOGS_VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    gpu="L40S",
    nonpreemptible=False,
    cpu=32,
    memory=131072,
    timeout=60 * 60 * 24,
    env={
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    },
    volumes={
        REMOTE_DATA: data_volume,
        REMOTE_CKPT: ckpt_volume,
        REMOTE_LOGS: logs_volume,
    },
)
def train_vgssm(train_args: str) -> str:
    workdir = Path(REMOTE_WORKDIR)
    src = Path(REMOTE_SRC)
    data_link = workdir / "data"
    ckpt_link = workdir / "checkpoints"
    logs_link = workdir / "logs"
    lightning_link = workdir / "lightning_logs"

    if workdir.exists():
        subprocess.run(["rm", "-rf", str(workdir)], check=True)
    workdir.mkdir(parents=True, exist_ok=True)

    subprocess.run(["rsync", "-a", "--delete", f"{src}/", f"{workdir}/"], check=True)

    data_candidates = [
        Path(REMOTE_DATA),
        Path(REMOTE_DATA) / "data",
        Path(REMOTE_DATA) / "data" / "data",
    ]
    data_target = Path(REMOTE_DATA)
    for candidate in data_candidates:
        if _looks_like_dataset_root(candidate):
            data_target = candidate
            break
    print(f"Using data root: {data_target}")

    for link_path, target in (
        (data_link, str(data_target)),
        (ckpt_link, REMOTE_CKPT),
        (logs_link, REMOTE_LOGS),
    ):
        if link_path.exists() or link_path.is_symlink():
            if link_path.is_dir() and not link_path.is_symlink():
                subprocess.run(["rm", "-rf", str(link_path)], check=True)
            else:
                link_path.unlink()
        os.symlink(target, str(link_path))

    if lightning_link.exists() or lightning_link.is_symlink():
        if lightning_link.is_dir() and not lightning_link.is_symlink():
            subprocess.run(["rm", "-rf", str(lightning_link)], check=True)
        else:
            lightning_link.unlink()
    lightning_link.mkdir(parents=True, exist_ok=True)

    os.chdir(workdir)
    train_tokens = shlex.split(train_args)

    mode = _get_cli_value(train_tokens, "--mode") or "train"

    # Preemption-safe resume only applies to training.
    # For prediction, always honor the explicit checkpoint.
    if mode == "predict":
        pred_ckpt = _get_cli_value(train_tokens, "--checkpoint")
        print(f"Prediction mode: using checkpoint={pred_ckpt}")
    else:
        resolved_ckpt = _resolve_resume_checkpoint(train_tokens)
        init_ckpt = _get_cli_value(train_tokens, "--init_ckpt")
        if resolved_ckpt is not None:
            _set_cli_value(train_tokens, "--checkpoint", resolved_ckpt)
            print(f"Resolved resume checkpoint: {resolved_ckpt}")
        elif _get_cli_value(train_tokens, "--checkpoint") is not None:
            user_ckpt = _get_cli_value(train_tokens, "--checkpoint")
            print(f"Checkpoint not found, keeping user arg as-is: {user_ckpt}")
        elif init_ckpt is not None:
            print(f"No resume checkpoint found; warm-starting from init_ckpt={init_ckpt}")
        else:
            print("No checkpoint provided; training will start from scratch.")

    cmd = ["python", "-u", "train_vgssm_standalone.py"] + train_tokens
    print("Running:", shlex.join(cmd))

    monitor_stop = threading.Event()

    def _gpu_monitor() -> None:
        while not monitor_stop.wait(GPU_MONITOR_INTERVAL_SEC):
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total",
                        "--format=csv,noheader",
                    ],
                    text=True,
                ).strip()
                if out:
                    for line in out.splitlines():
                        print(f"[gpu-monitor] {line}")
            except Exception as exc:
                print(f"[gpu-monitor] query failed: {exc}")

    monitor_thread = threading.Thread(target=_gpu_monitor, daemon=True)
    monitor_thread.start()

    run_error = None
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        run_error = exc
    finally:
        monitor_stop.set()
        monitor_thread.join(timeout=2.0)
        # Persist any generated submission artifacts.
        for submission_file in workdir.glob("submission*.parquet"):
            try:
                shutil.copy2(submission_file, Path(REMOTE_CKPT) / submission_file.name)
                print(f"Saved submission artifact: {submission_file.name}")
            except Exception as exc:
                print(f"Failed to persist submission artifact {submission_file.name}: {exc}")
        subprocess.run(
            ["rsync", "-a", f"{lightning_link}/", f"{REMOTE_LOGS}/lightning_logs_vgssm/"],
            check=False,
        )
        data_volume.commit()
        ckpt_volume.commit()
        logs_volume.commit()

    if run_error is not None:
        raise run_error

    return "training finished"


@app.local_entrypoint()
def main(train_args: str):
    result = train_vgssm.remote(train_args)
    print(result)
