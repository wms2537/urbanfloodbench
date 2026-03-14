"""
Modal launcher for UrbanFloodBench DualFlood checkpoint evaluation.

Usage:
  modal run modal_eval_dualflood.py --eval-args "<evaluate_dualflood.py args>"
"""

from __future__ import annotations

import os
import shlex
import subprocess
import threading
from pathlib import Path

import modal


APP_NAME = "urbanfloodbench-dualflood-eval"
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
    gpu="A100-80GB",
    cpu=16,
    memory=65536,
    timeout=60 * 60 * 8,
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
def eval_dualflood(eval_args: str) -> str:
    workdir = Path(REMOTE_WORKDIR)
    src = Path(REMOTE_SRC)
    data_link = workdir / "data"
    ckpt_link = workdir / "checkpoints"
    logs_link = workdir / "logs"

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
        if (candidate / "Model_1").exists() and (candidate / "Model_2").exists():
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

    os.chdir(workdir)
    cmd = ["python", "-u", "evaluate_dualflood.py"] + shlex.split(eval_args)
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
        data_volume.commit()
        ckpt_volume.commit()
        logs_volume.commit()

    if run_error is not None:
        raise run_error

    return "evaluation finished"


@app.local_entrypoint()
def main(eval_args: str):
    result = eval_dualflood.remote(eval_args)
    print(result)
