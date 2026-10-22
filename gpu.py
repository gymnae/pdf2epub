"""GPU probing and job planning.

Only NVML (or nvidia-smi) is used here, never torch, so the web process never
creates a CUDA context and holds zero VRAM while idle. All GPU work happens in
a short-lived subprocess that is fully torn down after each job.
"""
import os
import subprocess
from dataclasses import dataclass, field

GiB = 1024 ** 3


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


# Model weights + CUDA context of a marker 1.10 run. Not published upstream;
# check the "GPU peak" line in the logs after a run and adjust if needed.
BASE_VRAM_GB = _env_float("MARKER_BASE_VRAM_GB", 3.5)
# Headroom left free for whatever else uses the GPU.
SAFETY_MARGIN_GB = _env_float("GPU_SAFETY_MARGIN_GB", 1.0)
# Below this much working memory (after weights) a GPU run is not attempted.
MIN_WORK_VRAM_GB = _env_float("MARKER_MIN_WORK_VRAM_GB", 0.8)
# Multiplier for marker's own CUDA default batch sizes (upper bound).
BATCH_CAP_FACTOR = _env_float("MARKER_BATCH_CAP_FACTOR", 1.0)

# Per-item VRAM costs (GB) from the surya 0.17 README, and marker 1.10.2's CUDA
# defaults (marker passes these explicitly, so surya's *_BATCH_SIZE env vars
# have no effect - they must go through marker's config).
_BATCH_COSTS = {
    # config key: (GB per item, marker CUDA default)
    "layout_batch_size": (0.22, 12),
    "detection_batch_size": (0.44, 10),
    "recognition_batch_size": (0.04, 48),
    "equation_batch_size": (0.16, 32),  # recognition model, ~4x sequence length
    "table_rec_batch_size": (0.15, 14),
    "ocr_error_batch_size": (0.03, 14),
}


@dataclass
class GpuState:
    index: int
    name: str
    total: int
    free: int
    util: int = -1           # percent, -1 if unknown
    other_procs: int = -1    # compute processes already on the device, -1 if unknown

    @property
    def free_gb(self):
        return self.free / GiB

    @property
    def total_gb(self):
        return self.total / GiB

    def describe(self):
        s = f"GPU{self.index} {self.name}: {self.free_gb:.1f}/{self.total_gb:.1f} GiB free"
        if self.util >= 0:
            s += f", {self.util}% busy"
        if self.other_procs > 0:
            s += f", {self.other_procs} other process(es)"
        return s


@dataclass
class Plan:
    device: str                      # "cuda", "cpu" or "wait"
    gpu: GpuState = None
    budget_gb: float = 0.0
    config: dict = field(default_factory=dict)
    reason: str = ""


def _probe_nvml():
    import pynvml
    pynvml.nvmlInit()
    try:
        states = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode(errors="replace")
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
            except pynvml.NVMLError:
                util = -1
            try:
                # Other containers' PIDs are not resolvable here, but they are counted.
                procs = len(pynvml.nvmlDeviceGetComputeRunningProcesses(h))
            except pynvml.NVMLError:
                procs = -1
            states.append(GpuState(i, name, mem.total, mem.free, util, procs))
        return states
    finally:
        pynvml.nvmlShutdown()


def _probe_smi():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    states = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            util = int(parts[4])
        except ValueError:
            util = -1
        states.append(GpuState(int(parts[0]), parts[1], int(parts[2]) * 1024 ** 2,
                               int(parts[3]) * 1024 ** 2, util))
    return states


def probe():
    """Current state of all visible NVIDIA GPUs; empty list if none/unavailable."""
    try:
        return _probe_nvml()
    except Exception:
        return _probe_smi()


def used_bytes(index):
    """Used memory on one device (for peak tracking), or None."""
    for g in probe():
        if g.index == index:
            return g.total - g.free
    return None


def batch_config(work_gb):
    """Marker batch sizes that fit into `work_gb` of activation memory.

    Marker's stages run one after another, so each stage may use the whole
    working budget; the peak is weights + the largest stage, not the sum.
    """
    cfg = {}
    for key, (cost, default) in _BATCH_COSTS.items():
        cap = max(1, int(default * BATCH_CAP_FACTOR))
        cfg[key] = max(1, min(cap, int(work_gb / cost)))
    return cfg


def cpu_batch_config():
    # Small batches keep host RAM in check on the CPU path.
    return {"layout_batch_size": 2, "detection_batch_size": 2, "recognition_batch_size": 8,
            "equation_batch_size": 2, "table_rec_batch_size": 2, "ocr_error_batch_size": 4}


def plan_job(scale=1.0):
    """Decide where a conversion can run right now.

    `scale` < 1 shrinks all batch sizes (used for retries after an OOM).
    Returns a Plan with device "cuda" (with batch sizes), "wait" (a GPU exists
    but lacks free memory) or "cpu" (no GPU at all).
    """
    gpus = probe()
    if not gpus:
        return Plan("cpu", config=cpu_batch_config(), reason="no NVIDIA GPU visible")
    best = max(gpus, key=lambda g: g.free)
    work = best.free_gb - SAFETY_MARGIN_GB - BASE_VRAM_GB
    if work < MIN_WORK_VRAM_GB:
        need = BASE_VRAM_GB + MIN_WORK_VRAM_GB + SAFETY_MARGIN_GB
        return Plan("wait", best, reason=f"{best.describe()} (needs ≥ {need:.1f} GiB free)")
    cfg = {k: max(1, int(v * scale)) for k, v in batch_config(work).items()}
    note = f", batches ×{scale:g}" if scale < 1 else ""
    return Plan("cuda", best, work, cfg,
                reason=f"{best.describe()} → working budget {work:.1f} GiB{note}")
