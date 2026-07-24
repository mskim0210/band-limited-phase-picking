"""Step-level training profiler with GPU monitoring.

Decomposes each training step into four phases:
  I/O (DataLoader fetch) → Preprocess (CPU) → H2D (Host-to-Device) → Compute (GPU)

References:
    - MinatoLoader (EuroSys 2026): step time decomposition methodology
    - FFCV (Leclerc et al., CVPR 2023): TTA curve format
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# StepProfiler
# ---------------------------------------------------------------------------


class StepProfiler:
    """Decomposes each training step into four timed phases.

    Timing strategy:
        - I/O, Preprocess, H2D: ``time.perf_counter()`` (CPU-bound)
        - Compute: ``torch.cuda.Event`` (accurate GPU async measurement)
        - synchronize() is called only twice per step (start/end of compute)

    Args:
        warmup_steps: Number of initial steps excluded from summary stats.
        bytes_per_trace: Bytes for a single trace. Default 3*6000*4 = 72000.

    Example::

        profiler = StepProfiler(warmup_steps=10)
        for batch in dataloader:
            profiler.start_io()
            x, y = batch
            profiler.end_io()
            profiler.start_preprocess()
            x = normalize(x)
            profiler.end_preprocess()
            profiler.start_h2d()
            x, y = x.cuda(), y.cuda()
            profiler.end_h2d()
            profiler.start_compute()
            loss = model(x, y); loss.backward(); opt.step()
            profiler.end_compute()
        result = profiler.summary()
    """

    # Valid phase transitions: None → io → preprocess → h2d → compute → None
    _PHASE_ORDER = (None, "io", "preprocess", "h2d", "compute")

    def __init__(
        self,
        warmup_steps: int = 10,
        bytes_per_trace: int = 72_000,
    ) -> None:
        self._warmup_steps = warmup_steps
        self._bytes_per_trace = bytes_per_trace
        self._step_records: list[dict[str, Any]] = []
        self._current_step: dict[str, Any] = {}
        self._step_counter: int = 0
        self._batch_size: int = 0
        self._active_phase: str | None = None

        # Check CUDA availability once
        try:
            import torch.cuda

            self._use_cuda = torch.cuda.is_available()
        except ImportError:
            self._use_cuda = False

        if not self._use_cuda:
            logger.warning(
                "CUDA not available — compute timing will use perf_counter"
            )

    # -- Phase validation ----------------------------------------------------

    def _check_phase_order(self, expected_from: str | None, to: str) -> bool:
        """Validate that phases are called in the correct order.

        Args:
            expected_from: The phase that should currently be active (None for start).
            to: The phase being transitioned to.

        Returns:
            True if order is valid, False otherwise (with warning logged).
        """
        if self._active_phase != expected_from:
            logger.warning(
                "Step %d: phase order error — expected '%s' active, got '%s'. "
                "Skipping transition to '%s'.",
                self._step_counter,
                expected_from,
                self._active_phase,
                to,
            )
            return False
        return True

    # -- I/O phase ----------------------------------------------------------

    def start_io(self) -> None:
        """Mark the beginning of the I/O (DataLoader fetch) phase."""
        if not self._check_phase_order(None, "io"):
            return
        self._active_phase = "io"
        self._current_step = {"step": self._step_counter}
        self._io_start = time.perf_counter()

    def end_io(self, batch_size: int = 0) -> None:
        """Mark the end of the I/O phase.

        Args:
            batch_size: Actual batch size (for bytes_read calculation).
                If 0, bytes_read will be 0 for this step.
        """
        if not self._check_phase_order("io", "io_end"):
            return
        self._active_phase = None
        self._current_step["t_io"] = time.perf_counter() - self._io_start
        self._batch_size = batch_size

    # -- Preprocess phase ---------------------------------------------------

    def start_preprocess(self) -> None:
        """Mark the beginning of the CPU preprocessing phase."""
        self._active_phase = "preprocess"
        self._preprocess_start = time.perf_counter()

    def end_preprocess(self) -> None:
        """Mark the end of the preprocessing phase."""
        if not self._check_phase_order("preprocess", "preprocess_end"):
            return
        self._active_phase = None
        self._current_step["t_preprocess"] = (
            time.perf_counter() - self._preprocess_start
        )

    # -- H2D phase ----------------------------------------------------------

    def start_h2d(self) -> None:
        """Mark the beginning of the Host-to-Device transfer phase."""
        self._active_phase = "h2d"
        self._h2d_start = time.perf_counter()

    def end_h2d(self) -> None:
        """Mark the end of the H2D phase."""
        if not self._check_phase_order("h2d", "h2d_end"):
            return
        self._active_phase = None
        self._current_step["t_h2d"] = time.perf_counter() - self._h2d_start

    # -- Compute phase ------------------------------------------------------

    def start_compute(self) -> None:
        """Mark the beginning of the GPU compute phase.

        Calls ``torch.cuda.synchronize()`` to ensure prior async ops
        (e.g. H2D transfers) have completed before timing starts.
        """
        self._active_phase = "compute"
        if self._use_cuda:
            import torch.cuda

            torch.cuda.synchronize()
            self._compute_start_event = torch.cuda.Event(enable_timing=True)
            self._compute_end_event = torch.cuda.Event(enable_timing=True)
            self._compute_start_event.record()
        else:
            self._compute_start_cpu = time.perf_counter()

    def end_compute(self) -> None:
        """Mark the end of the GPU compute phase and finalize the step record.

        Calls ``torch.cuda.synchronize()`` and uses CUDA event elapsed time
        for accurate GPU measurement.
        """
        if not self._check_phase_order("compute", "compute_end"):
            return
        self._active_phase = None

        if self._use_cuda:
            import torch.cuda

            self._compute_end_event.record()
            torch.cuda.synchronize()
            # elapsed_time() returns milliseconds
            t_compute = self._compute_start_event.elapsed_time(
                self._compute_end_event
            ) / 1000.0
        else:
            t_compute = time.perf_counter() - self._compute_start_cpu

        self._current_step["t_compute"] = t_compute

        # Compute total and bytes_read
        t_total = (
            self._current_step.get("t_io", 0.0)
            + self._current_step.get("t_preprocess", 0.0)
            + self._current_step.get("t_h2d", 0.0)
            + t_compute
        )
        self._current_step["t_total"] = t_total
        self._current_step["bytes_read"] = (
            self._batch_size * self._bytes_per_trace
        )

        self._step_records.append(self._current_step)
        self._step_counter += 1
        self._current_step = {}

    # -- Summary & utilities ------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Compute summary statistics over recorded steps (excluding warmup).

        Returns:
            Dictionary with keys:
                - ``step_times``: All step records (including warmup).
                - ``summary``: Per-phase stats (mean, p50, p95, total),
                  plus throughput and step counts.
        """
        all_records = self._step_records
        # Exclude warmup for statistics
        records = all_records[self._warmup_steps:]

        if not records:
            logger.warning(
                "No steps after warmup (total=%d, warmup=%d)",
                len(all_records),
                self._warmup_steps,
            )
            return {"step_times": all_records, "summary": {}}

        phase_keys = ["t_io", "t_preprocess", "t_h2d", "t_compute", "t_total"]
        summary: dict[str, Any] = {}

        for key in phase_keys:
            values = np.array([r.get(key, 0.0) for r in records])
            summary[key] = {
                "mean": float(np.mean(values)),
                "p50": float(np.median(values)),
                "p95": float(np.percentile(values, 95)),
                "total": float(np.sum(values)),
            }

        # Throughput: traces / total wall-clock time (after warmup)
        total_time = summary["t_total"]["total"]
        total_bytes = sum(r.get("bytes_read", 0) for r in records)
        total_traces = total_bytes / self._bytes_per_trace if self._bytes_per_trace > 0 else 0

        summary["throughput_traces_per_sec"] = (
            total_traces / total_time if total_time > 0 else 0.0
        )
        summary["num_steps"] = len(records)
        summary["num_warmup_steps"] = self._warmup_steps

        return {"step_times": all_records, "summary": summary}

    def reset(self) -> None:
        """Clear all recorded data for a new profiling session."""
        self._step_records.clear()
        self._current_step = {}
        self._step_counter = 0
        self._batch_size = 0
        self._active_phase = None


# ---------------------------------------------------------------------------
# GPUMonitor
# ---------------------------------------------------------------------------


class GPUMonitor:
    """Collects GPU utilization via nvidia-smi in a background thread.

    Args:
        interval: Polling interval in seconds.
        gpu_index: GPU index to monitor.

    Example::

        monitor = GPUMonitor(interval=1.0)
        monitor.start()
        # ... training loop ...
        gpu_records = monitor.stop()
    """

    def __init__(
        self,
        interval: float = 1.0,
        gpu_index: int = 0,
    ) -> None:
        self._interval = interval
        self._gpu_index = gpu_index
        self._records: list[dict[str, Any]] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background monitoring thread."""
        self._stop_event.clear()
        self._records.clear()
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="gpu-monitor"
        )
        self._thread.start()
        logger.info(
            "GPUMonitor started (interval=%.1fs, gpu=%d)",
            self._interval,
            self._gpu_index,
        )

    def stop(self) -> list[dict[str, Any]]:
        """Stop monitoring and return collected records.

        Returns:
            List of GPU state dictionaries with keys:
            ``timestamp``, ``gpu_util_pct``, ``mem_util_pct``,
            ``mem_used_mb``, ``temperature_c``.
        """
        if self._thread is None:
            logger.warning("GPUMonitor.stop() called without start()")
            return []

        self._stop_event.set()
        self._thread.join(timeout=self._interval + 2.0)
        self._thread = None
        logger.info("GPUMonitor stopped — %d records collected", len(self._records))
        return list(self._records)

    def _poll_loop(self) -> None:
        """Internal polling loop running in a daemon thread."""
        while not self._stop_event.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--query-gpu=utilization.gpu,utilization.memory,"
                        f"memory.used,temperature.gpu",
                        "--format=csv,noheader,nounits",
                        f"-i", str(self._gpu_index),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5.0,
                )
                if result.returncode == 0:
                    record = self._parse_nvidia_smi(result.stdout)
                    if record:
                        record["timestamp"] = datetime.now(timezone.utc).isoformat()
                        self._records.append(record)
                else:
                    logger.warning(
                        "nvidia-smi returned code %d: %s",
                        result.returncode,
                        result.stderr.strip(),
                    )
            except FileNotFoundError:
                logger.warning("nvidia-smi not found — GPU monitoring disabled")
                break
            except subprocess.TimeoutExpired:
                logger.warning("nvidia-smi timed out")
            except Exception:
                logger.exception("Unexpected error in GPU monitor")

            self._stop_event.wait(self._interval)

    @staticmethod
    def _parse_nvidia_smi(output: str) -> dict[str, Any]:
        """Parse nvidia-smi CSV output into a dictionary.

        Args:
            output: Raw stdout from nvidia-smi, e.g. ``"85, 42, 3940, 45\\n"``.

        Returns:
            Dictionary with gpu_util_pct, mem_util_pct, mem_used_mb,
            temperature_c. Empty dict on parse failure.
        """
        try:
            parts = [p.strip() for p in output.strip().split(",")]
            if len(parts) < 4:
                logger.warning("nvidia-smi output has < 4 fields: %r", output)
                return {}
            return {
                "gpu_util_pct": float(parts[0]),
                "mem_util_pct": float(parts[1]),
                "mem_used_mb": int(float(parts[2])),
                "temperature_c": int(float(parts[3])),
            }
        except (ValueError, IndexError):
            logger.warning("Failed to parse nvidia-smi output: %r", output)
            return {}


# ---------------------------------------------------------------------------
# save_profile
# ---------------------------------------------------------------------------


def save_profile(
    step_result: dict[str, Any],
    gpu_records: list[dict[str, Any]],
    metadata: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Serialize profiling results to a JSON file.

    Args:
        step_result: Return value of ``StepProfiler.summary()``.
        gpu_records: Return value of ``GPUMonitor.stop()``.
        metadata: Run metadata (level, dataset, batch_size, etc.).
        output_path: Destination file path.

    Returns:
        Path object of the saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute GPU summary from records
    gpu_summary: dict[str, float] = {}
    if gpu_records:
        gpu_utils = [r["gpu_util_pct"] for r in gpu_records if "gpu_util_pct" in r]
        mem_utils = [r["mem_util_pct"] for r in gpu_records if "mem_util_pct" in r]
        if gpu_utils:
            gpu_summary["gpu_util_mean_pct"] = float(np.mean(gpu_utils))
        if mem_utils:
            gpu_summary["gpu_mem_mean_pct"] = float(np.mean(mem_utils))

    # Merge GPU summary into step summary
    summary = step_result.get("summary", {})
    summary.update(gpu_summary)

    profile = {
        "metadata": metadata,
        "step_times": step_result.get("step_times", []),
        "gpu_log": gpu_records,
        "summary": summary,
    }

    with open(output_path, "w") as f:
        json.dump(profile, f, indent=2, default=str)

    logger.info("Profile saved to %s", output_path)
    return output_path
