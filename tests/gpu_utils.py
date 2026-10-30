"""
GPU selection for the GPU checks (tests/run_checks.py, tests/check_api.py), which run on a machine shared with other
jobs: instead of fixed GPU ids, the GPUs free at that moment are taken right before each job, and the job is skipped if
there are none, so it neither crashes for lack of memory nor slows down (or crashes) the jobs already running.
"""
import subprocess
from typing import List, Optional

# A GPU with this many MiB in use or more is considered busy (an idle GPU shows a few MiB)
DEFAULT_MAX_USED_MEMORY_MB = 500


def free_gpus(candidates: Optional[List[str]] = None, max_used_memory_mb: int = DEFAULT_MAX_USED_MEMORY_MB) -> List[str]:
    """
    GPUs with no process running on them and less than ``max_used_memory_mb`` MiB in use.

    Parameters
    ----------
    candidates : List[str], optional
        GPU ids (as ``nvidia-smi`` numbers them) to choose from. All the GPUs of the machine if not given.

    max_used_memory_mb : int, optional
        Memory in use (MiB) from which a GPU is considered busy. It also catches the processes ``nvidia-smi`` can not
        list (e.g. from other containers).

    Returns
    -------
    gpus : List[str]
        Ids of the free GPUs, in ``nvidia-smi`` order (the one BiaPy uses, as it sets 'CUDA_DEVICE_ORDER=PCI_BUS_ID').
        Empty if there is no GPU or ``nvidia-smi`` is not available.
    """
    try:
        gpus = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"Could not query the GPUs with nvidia-smi: {e}")
        return []

    busy_uuids = {line.strip() for line in apps.splitlines() if line.strip()}
    free = []
    for line in gpus.splitlines():
        if not line.strip():
            continue
        index, uuid, used = [x.strip() for x in line.split(",")]
        if candidates is not None and index not in candidates:
            continue
        if uuid in busy_uuids or float(used) >= max_used_memory_mb:
            continue
        free.append(index)
    return free


def gpu_report() -> str:
    """Use of every GPU, to explain why a job was skipped."""
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as e:
        return f"nvidia-smi not available: {e}"
