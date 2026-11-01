"""Synthetic correctness and overhead experiment for Phase 7 secure aggregation."""

from __future__ import annotations

import io
import json
import secrets
import statistics
import sys
import time
import unittest
from pathlib import Path
from typing import Dict, Tuple

import torch

from server.aggregator import fedavg
from server.secure_aggregation import (
    SecureAggregationConfig,
    SecureAggregationParticipant,
    SecureAggregationProtocol,
)


def _new_secure_round(
    client_ids: Tuple[int, ...],
    config: SecureAggregationConfig,
):
    protocol = SecureAggregationProtocol(config, client_ids=client_ids)
    participants = {
        client_id: SecureAggregationParticipant(client_id, config)
        for client_id in client_ids
    }
    round_id = secrets.token_hex(16)
    public_keys = {
        client_id: participant.begin_round(round_id, client_ids)
        for client_id, participant in participants.items()
    }
    for participant in participants.values():
        participant.accept_public_keys(round_id, public_keys)
    for participant in participants.values():
        participant.distribute_recovery_shares(
            participants, int(config.recovery_threshold)
        )
    protocol.begin_round(round_id, public_keys)
    return protocol, participants


def benchmark_secure_aggregation(
    client_count: int = 5,
    tensor_elements: int = 4_096,
    repetitions: int = 5,
) -> Dict[str, object]:
    """Compare synthetic weighted FedAvg with the secure masked-sum path."""
    client_ids = tuple(range(1, client_count + 1))
    config = SecureAggregationConfig(
        enabled=True,
        quantization_scale=10_000.0,
        modulus=2**61 - 1,
        min_clients=2,
        max_dropout_fraction=0.0,
        max_abs_update=1.0,
    )
    coordinates = torch.arange(tensor_elements, dtype=torch.float32)
    states: Dict[int, Dict[str, torch.Tensor]] = {
        client_id: {
            "synthetic_weight": 0.25
            * torch.sin(coordinates / 83.0 + client_id / 7.0),
            "synthetic_bias": torch.tensor(
                [client_id / 100.0, -client_id / 200.0], dtype=torch.float32
            ),
        }
        for client_id in client_ids
    }
    sample_counts = {
        client_id: 100 + 11 * client_id for client_id in client_ids
    }

    standard_durations = []
    standard_result = None
    for _ in range(repetitions):
        started = time.perf_counter()
        standard_result = fedavg(
            [(states[client_id], sample_counts[client_id]) for client_id in client_ids]
        )
        standard_durations.append(time.perf_counter() - started)

    secure_durations = []
    secure_result = None
    for _ in range(repetitions):
        started = time.perf_counter()
        protocol, participants = _new_secure_round(client_ids, config)
        masked_updates = [
            participants[client_id].mask_client_update(
                states[client_id], sample_counts[client_id]
            )
            for client_id in client_ids
        ]
        secure_result = protocol.aggregate(masked_updates)
        secure_durations.append(time.perf_counter() - started)

    max_error = max(
        (secure_result[key] - standard_result[key]).abs().max().item()
        for key in standard_result
    )
    if max_error > 0.5 / config.quantization_scale + 1e-7:
        raise AssertionError(
            f"Secure aggregation quantization error {max_error} exceeds its bound."
        )

    state_bytes_per_client = sum(
        tensor.numel() * tensor.element_size()
        for tensor in states[client_ids[0]].values()
    )
    standard_upload_bytes = state_bytes_per_client * client_count
    secure_upload_bytes = (tensor_elements + 2) * 8 * client_count
    recovery_share_bytes = 68 * client_count * (client_count - 1)
    public_key_bytes = 32 * client_count
    communication_setup_bytes = recovery_share_bytes + public_key_bytes
    standard_median = statistics.median(standard_durations)
    secure_median = statistics.median(secure_durations)

    return {
        "protocol": "Bonawitz-style pairwise masks (X25519 + Shamir dropout recovery)",
        "synthetic_only": True,
        "client_count": client_count,
        "tensor_elements_per_client": tensor_elements + 2,
        "repetitions": repetitions,
        "sample_counts": sample_counts,
        "fedavg_median_seconds": standard_median,
        "secure_fedavg_median_seconds": secure_median,
        "measured_compute_overhead_percent": (
            100.0 * (secure_median / standard_median - 1.0)
            if standard_median
            else None
        ),
        "max_absolute_quantization_error": max_error,
        "quantization_error_bound": 0.5 / config.quantization_scale,
        "standard_tensor_upload_bytes": standard_upload_bytes,
        "secure_tensor_upload_bytes": secure_upload_bytes,
        "secure_tensor_payload_overhead_percent": (
            100.0 * (secure_upload_bytes / standard_upload_bytes - 1.0)
        ),
        "estimated_public_key_and_share_bytes": communication_setup_bytes,
        "estimated_secure_round_bytes_including_setup": (
            secure_upload_bytes + communication_setup_bytes
        ),
        "communication_estimate_excludes": [
            "transport framing",
            "round IDs and tensor metadata",
            "TLS/authentication overhead",
            "dropout recovery messages",
        ],
    }


def _run_secure_aggregation_tests(output_dir: Path) -> Dict[str, object]:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(
        str(project_root / "tests"),
        pattern="test_secure_aggregation.py",
    )
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    test_report = stream.getvalue()
    (output_dir / "secure_aggregation_test_results.txt").write_text(
        test_report, encoding="utf-8"
    )
    summary = {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "successful": result.wasSuccessful(),
        "details_file": "secure_aggregation_test_results.txt",
    }
    if not result.wasSuccessful():
        raise AssertionError(
            f"Secure aggregation tests failed ({summary}); see "
            f"{output_dir / 'secure_aggregation_test_results.txt'}."
        )
    return summary


def _run_full_test_suite(output_dir: Path) -> Dict[str, object]:
    project_root = Path(__file__).resolve().parents[1]
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(
        str(project_root / "tests"),
        pattern="test_*.py",
    )
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    test_report = stream.getvalue()
    (output_dir / "full_test_suite.txt").write_text(test_report, encoding="utf-8")
    summary = {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "successful": result.wasSuccessful(),
        "details_file": "full_test_suite.txt",
    }
    if not result.wasSuccessful():
        raise AssertionError(
            f"The full test suite failed ({summary}); see "
            f"{output_dir / 'full_test_suite.txt'}."
        )
    return summary


def run_phase7(output_dir: str | Path = "results/phase7") -> Dict[str, object]:
    """Run focused tests and a synthetic benchmark without loading a dataset."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    secure_tests = _run_secure_aggregation_tests(destination)
    full_tests = _run_full_test_suite(destination)
    benchmark = benchmark_secure_aggregation()
    results = {
        "phase": 7,
        "tests": {
            "secure_aggregation": secure_tests,
            "full_suite": full_tests,
        },
        "benchmark": benchmark,
        "security_assumptions": [
            "The server and at most recovery_threshold - 1 surviving clients do not collude to reconstruct a non-uploader key.",
            "At least two non-colluding uploading clients remain for an aggregate to hide individual updates.",
            "The simulator's in-memory client-to-client share exchange stands in for authenticated private channels.",
        ],
        "limitations": [
            "No network transport, public-key authentication, secure hardware, or production deployment is implemented.",
            "The server learns sample counts and existing per-client loss/accuracy monitoring scalars.",
            "Integer state buffers are sample-weight averaged and rounded; the existing FedAvg path uses max for discrete buffers. Equal BatchNorm counters remain unchanged.",
            "Secure aggregation hides individual updates only within a round; repeated or differenced aggregates can still disclose information.",
            "Local DP followed by secure aggregation is supported, but DP guarantees depend on the existing accountant and per-client noise configuration.",
        ],
    }
    (destination / "phase7_benchmark.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    report = f"""# Phase 7: Secure Aggregation Results

## Protocol and architecture

The implementation follows the pairwise-mask design in Bonawitz et al.,
“Practical Secure Aggregation for Privacy-Preserving Machine Learning”
(ACM CCS 2017). Each client creates a fresh X25519 key pair per round.
X25519 shared secrets are expanded with HKDF-SHA-256 and HMAC-SHA-256 into
opposite-sign masks. Clients fixed-point quantize and sample-weight updates
before masking; the server sees and sums only masked modular tensors.

If a client fails to upload, active clients supply Shamir shares of that
non-uploader's ephemeral private key. The server reconstructs only that key,
removes the unmatched masks, and aborts if the threshold, minimum participation,
or dropout budget is not met. Masked uploads are round-bound and validated for
unique IDs, expected keys/shapes/dtypes, positive sample counts, modular range,
and bounded aggregate magnitude.

## Synthetic comparison and overhead

- Clients: {benchmark['client_count']}; entries per client: {benchmark['tensor_elements_per_client']:,}; repetitions: {benchmark['repetitions']}.
- Ordinary weighted FedAvg median: {benchmark['fedavg_median_seconds']:.6f} seconds.
- Secure FedAvg median (key setup, masking, and aggregate): {benchmark['secure_fedavg_median_seconds']:.6f} seconds.
- Measured computation overhead: {benchmark['measured_compute_overhead_percent']:.1f}% on this synthetic CPU run.
- Plain tensor upload: {benchmark['standard_tensor_upload_bytes']:,} bytes; secure masked tensor upload: {benchmark['secure_tensor_upload_bytes']:,} bytes ({benchmark['secure_tensor_payload_overhead_percent']:.1f}% tensor-payload overhead).
- Estimated round key/share setup traffic: {benchmark['estimated_public_key_and_share_bytes']:,} bytes, excluding metadata and transport framing.
- Maximum observed difference from FedAvg: {benchmark['max_absolute_quantization_error']:.8f}; configured bound: {benchmark['quantization_error_bound']:.8f}.

These are microbenchmark measurements on synthetic tensors, not model-training
latency or a network benchmark. Use `phase7_benchmark.json` for machine-readable
measurements. Detailed secure aggregation and full-suite test logs are in
`secure_aggregation_test_results.txt` ({secure_tests['tests_run']} passed) and
`full_test_suite.txt` ({full_tests['tests_run']} passed).

## Threat model and limitations

- The goal is to prevent the honest-but-curious aggregation server from seeing
  any single uploading client's plaintext update, while revealing the weighted
  sum to it.
- Pairwise public keys are not authenticated in this simulation. Real clients
  need authenticated identities, private channels for share exchange, replay
  protection, and integrity checks.
- Recovery assumes the server plus fewer than the Shamir threshold surviving
  clients cannot collude. A recovered dropout key is intentionally exposed to
  the server for that round. At least two honest uploaders are necessary;
  one colluding uploader can reveal another update if only two clients remain.
- In-memory client objects do not provide process isolation. This validates
  protocol arithmetic and control flow, not deployment security.
- Updates above `max_abs_update`, or counts that can wrap the signed modular
  sum, are rejected. Fixed-point quantization introduces at most half a step
  of per-coordinate error in the aggregate average.
- Integer model buffers are decoded as rounded weighted averages in secure
  mode. This matches the usual identical BatchNorm counters, but not FedAvg's
  max rule for arbitrary differing integer buffers.
- Secure aggregation is not differential privacy. `secure_dp` applies the
  existing client-side DP-SGD before masking; its accounting and monitoring
  outputs retain their existing caveats.

## Test results

Secure aggregation tests: {secure_tests['tests_run']} run, {secure_tests['failures']} failures,
{secure_tests['errors']} errors. Full suite: {full_tests['tests_run']} run,
{full_tests['failures']} failures, {full_tests['errors']} errors.
"""
    (destination / "phase7_report.md").write_text(report, encoding="utf-8")
    return results


if __name__ == "__main__":
    output = run_phase7()
    print(json.dumps(output, indent=2))
