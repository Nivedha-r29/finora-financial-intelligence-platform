# Phase 7: Secure Aggregation Results

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

- Clients: 5; entries per client: 4,098; repetitions: 5.
- Ordinary weighted FedAvg median: 0.001179 seconds.
- Secure FedAvg median (key setup, masking, and aggregate): 0.512600 seconds.
- Measured computation overhead: 43366.4% on this synthetic CPU run.
- Plain tensor upload: 81,960 bytes; secure masked tensor upload: 163,920 bytes (100.0% tensor-payload overhead).
- Estimated round key/share setup traffic: 1,520 bytes, excluding metadata and transport framing.
- Maximum observed difference from FedAvg: 0.00004128; configured bound: 0.00005000.

These are microbenchmark measurements on synthetic tensors, not model-training
latency or a network benchmark. Use `phase7_benchmark.json` for machine-readable
measurements. Detailed secure aggregation and full-suite test logs are in
`secure_aggregation_test_results.txt` (18 passed) and
`full_test_suite.txt` (34 passed).

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

Secure aggregation tests: 18 run, 0 failures,
0 errors. Full suite: 34 run,
0 failures, 0 errors.
