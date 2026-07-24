"""Unit tests for cs_pipeline.profiler module."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from cs_pipeline.profiler import GPUMonitor, StepProfiler, save_profile


# ---------------------------------------------------------------------------
# StepProfiler tests
# ---------------------------------------------------------------------------


class TestStepProfiler:
    """Tests for StepProfiler class."""

    def test_single_step_records_four_phases(self) -> None:
        """One step should produce a record with all four phase timings."""
        profiler = StepProfiler(warmup_steps=0)

        profiler.start_io()
        time.sleep(0.01)
        profiler.end_io(batch_size=32)

        profiler.start_preprocess()
        time.sleep(0.005)
        profiler.end_preprocess()

        profiler.start_h2d()
        time.sleep(0.002)
        profiler.end_h2d()

        profiler.start_compute()
        time.sleep(0.01)
        profiler.end_compute()

        result = profiler.summary()
        assert len(result["step_times"]) == 1

        record = result["step_times"][0]
        for key in ("t_io", "t_preprocess", "t_h2d", "t_compute", "t_total"):
            assert key in record, f"Missing key: {key}"
            assert record[key] > 0, f"{key} should be > 0"

    def test_total_equals_sum_of_phases(self) -> None:
        """t_total should approximately equal the sum of four phases."""
        profiler = StepProfiler(warmup_steps=0)

        profiler.start_io()
        time.sleep(0.01)
        profiler.end_io(batch_size=16)

        profiler.start_preprocess()
        time.sleep(0.005)
        profiler.end_preprocess()

        profiler.start_h2d()
        time.sleep(0.002)
        profiler.end_h2d()

        profiler.start_compute()
        time.sleep(0.01)
        profiler.end_compute()

        record = profiler.summary()["step_times"][0]
        expected = (
            record["t_io"]
            + record["t_preprocess"]
            + record["t_h2d"]
            + record["t_compute"]
        )
        assert abs(record["t_total"] - expected) < 0.001

    def test_warmup_excluded_from_summary(self) -> None:
        """Steps within warmup should be excluded from summary statistics."""
        profiler = StepProfiler(warmup_steps=2)

        # Run 5 steps
        for _ in range(5):
            profiler.start_io()
            profiler.end_io(batch_size=8)
            profiler.start_preprocess()
            profiler.end_preprocess()
            profiler.start_h2d()
            profiler.end_h2d()
            profiler.start_compute()
            profiler.end_compute()

        result = profiler.summary()
        assert len(result["step_times"]) == 5  # all steps recorded
        assert result["summary"]["num_steps"] == 3  # 5 - 2 warmup
        assert result["summary"]["num_warmup_steps"] == 2

    def test_summary_statistics(self) -> None:
        """mean, p50, p95, total should be correctly computed."""
        profiler = StepProfiler(warmup_steps=0, bytes_per_trace=100)

        # Run 3 steps with controlled sleep for approximate timing
        for _ in range(3):
            profiler.start_io()
            time.sleep(0.01)
            profiler.end_io(batch_size=10)
            profiler.start_preprocess()
            profiler.end_preprocess()
            profiler.start_h2d()
            profiler.end_h2d()
            profiler.start_compute()
            profiler.end_compute()

        result = profiler.summary()
        summary = result["summary"]

        # t_io should have valid stats
        assert summary["t_io"]["mean"] > 0
        assert summary["t_io"]["p50"] > 0
        assert summary["t_io"]["p95"] >= summary["t_io"]["p50"]
        assert summary["t_io"]["total"] > 0

        # Throughput should be > 0
        assert summary["throughput_traces_per_sec"] > 0

    def test_bytes_read_calculation(self) -> None:
        """bytes_read should equal batch_size * bytes_per_trace."""
        profiler = StepProfiler(warmup_steps=0, bytes_per_trace=72_000)

        profiler.start_io()
        profiler.end_io(batch_size=64)
        profiler.start_preprocess()
        profiler.end_preprocess()
        profiler.start_h2d()
        profiler.end_h2d()
        profiler.start_compute()
        profiler.end_compute()

        record = profiler.summary()["step_times"][0]
        assert record["bytes_read"] == 64 * 72_000

    def test_reset_clears_state(self) -> None:
        """reset() should clear all recorded data."""
        profiler = StepProfiler(warmup_steps=0)

        profiler.start_io()
        profiler.end_io(batch_size=1)
        profiler.start_preprocess()
        profiler.end_preprocess()
        profiler.start_h2d()
        profiler.end_h2d()
        profiler.start_compute()
        profiler.end_compute()

        assert len(profiler.summary()["step_times"]) == 1

        profiler.reset()
        assert len(profiler.summary()["step_times"]) == 0

    def test_no_cuda_fallback(self) -> None:
        """When CUDA is not available, compute should use perf_counter."""
        profiler = StepProfiler(warmup_steps=0)
        profiler._use_cuda = False  # Force CPU fallback

        profiler.start_io()
        profiler.end_io(batch_size=1)
        profiler.start_preprocess()
        profiler.end_preprocess()
        profiler.start_h2d()
        profiler.end_h2d()
        profiler.start_compute()
        time.sleep(0.005)
        profiler.end_compute()

        record = profiler.summary()["step_times"][0]
        assert record["t_compute"] > 0


# ---------------------------------------------------------------------------
# GPUMonitor tests
# ---------------------------------------------------------------------------


class TestGPUMonitor:
    """Tests for GPUMonitor class."""

    def test_parse_nvidia_smi_valid(self) -> None:
        """Valid nvidia-smi output should parse correctly."""
        output = "85, 42, 3940, 45\n"
        result = GPUMonitor._parse_nvidia_smi(output)

        assert result["gpu_util_pct"] == 85.0
        assert result["mem_util_pct"] == 42.0
        assert result["mem_used_mb"] == 3940
        assert result["temperature_c"] == 45

    def test_parse_nvidia_smi_with_spaces(self) -> None:
        """nvidia-smi output with varying whitespace should parse."""
        output = " 12 , 8 , 1024 , 38 \n"
        result = GPUMonitor._parse_nvidia_smi(output)

        assert result["gpu_util_pct"] == 12.0
        assert result["mem_used_mb"] == 1024

    def test_parse_nvidia_smi_malformed(self) -> None:
        """Malformed output should return empty dict."""
        result = GPUMonitor._parse_nvidia_smi("garbage data")
        assert result == {}

    def test_parse_nvidia_smi_too_few_fields(self) -> None:
        """Output with fewer than 4 fields should return empty dict."""
        result = GPUMonitor._parse_nvidia_smi("85, 42\n")
        assert result == {}

    def test_stop_without_start(self) -> None:
        """stop() without start() should return empty list."""
        monitor = GPUMonitor()
        result = monitor.stop()
        assert result == []

    def test_start_stop_returns_records(self) -> None:
        """start() then stop() after a short delay should collect records."""
        mock_output = "50, 30, 2048, 42\n"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = type(
                "Result", (), {"returncode": 0, "stdout": mock_output, "stderr": ""}
            )()

            monitor = GPUMonitor(interval=0.1)
            monitor.start()
            time.sleep(0.35)
            records = monitor.stop()

            assert len(records) >= 1
            assert records[0]["gpu_util_pct"] == 50.0
            assert "timestamp" in records[0]


# ---------------------------------------------------------------------------
# save_profile tests
# ---------------------------------------------------------------------------


class TestSaveProfile:
    """Tests for save_profile function."""

    def test_json_schema(self, tmp_path: Path) -> None:
        """Saved JSON should contain all required top-level keys."""
        step_result = {
            "step_times": [
                {
                    "step": 0,
                    "t_io": 0.5,
                    "t_preprocess": 0.01,
                    "t_h2d": 0.001,
                    "t_compute": 0.05,
                    "t_total": 0.561,
                    "bytes_read": 72000,
                }
            ],
            "summary": {
                "t_io": {"mean": 0.5, "p50": 0.5, "p95": 0.5, "total": 0.5},
                "throughput_traces_per_sec": 100.0,
            },
        }
        gpu_records = [
            {
                "timestamp": "2026-03-26T00:00:00+00:00",
                "gpu_util_pct": 11.6,
                "mem_util_pct": 11.7,
                "mem_used_mb": 3940,
                "temperature_c": 45,
            }
        ]
        metadata = {"level": "L0", "batch_size": 64}

        out = tmp_path / "test_profile.json"
        result_path = save_profile(step_result, gpu_records, metadata, out)

        assert result_path.exists()
        with open(result_path) as f:
            data = json.load(f)

        assert "metadata" in data
        assert "step_times" in data
        assert "gpu_log" in data
        assert "summary" in data

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        """Should create parent directories if they don't exist."""
        out = tmp_path / "nested" / "dir" / "profile.json"

        save_profile(
            {"step_times": [], "summary": {}}, [], {"level": "test"}, out
        )

        assert out.exists()

    def test_summary_includes_gpu_stats(self, tmp_path: Path) -> None:
        """Summary should include gpu_util_mean_pct and gpu_mem_mean_pct."""
        step_result = {"step_times": [], "summary": {"t_io": {"mean": 0.1}}}
        gpu_records = [
            {"gpu_util_pct": 80.0, "mem_util_pct": 50.0},
            {"gpu_util_pct": 90.0, "mem_util_pct": 60.0},
        ]

        out = tmp_path / "profile.json"
        save_profile(step_result, gpu_records, {"level": "L2"}, out)

        with open(out) as f:
            data = json.load(f)

        assert data["summary"]["gpu_util_mean_pct"] == pytest.approx(85.0)
        assert data["summary"]["gpu_mem_mean_pct"] == pytest.approx(55.0)
