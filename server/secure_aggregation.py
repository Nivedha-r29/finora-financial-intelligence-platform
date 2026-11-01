"""Pairwise-mask secure aggregation based on Bonawitz et al. (CCS 2017).

Client updates are fixed-point encoded, sample-weighted, and masked with
pairwise X25519/HKDF-derived pseudorandom vectors. Pairwise masks cancel in
the sum. Shamir shares let the server recover only the private key of a client
that did not upload, so masks on the surviving clients' updates can be removed.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import secrets
from dataclasses import dataclass
from numbers import Integral
from typing import Dict, Iterable, Mapping, Sequence, Set, Tuple

import numpy as np
import torch
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


_SHAMIR_FIELD = (1 << 521) - 1
_SUPPORTED_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


@dataclass
class SecureAggregationConfig:
    """Bounds and participation policy for one secure aggregation setup."""

    enabled: bool = False
    quantization_scale: float = 1_000.0
    modulus: int = 2**61 - 1
    min_clients: int = 2
    max_dropout_fraction: float = 0.2
    allow_recovery: bool = True
    recovery_threshold: int | None = None
    max_abs_update: float = 1_000_000.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.quantization_scale) or self.quantization_scale <= 0:
            raise ValueError("quantization_scale must be finite and positive.")
        if (
            not isinstance(self.modulus, Integral)
            or isinstance(self.modulus, bool)
            or self.modulus < 3
            or self.modulus % 2 == 0
            or self.modulus > 2**62 - 1
        ):
            raise ValueError("modulus must be an odd integer in [3, 2**62 - 1].")
        if not isinstance(self.min_clients, Integral) or isinstance(self.min_clients, bool) or self.min_clients < 2:
            raise ValueError("min_clients must be an integer of at least 2.")
        if not math.isfinite(self.max_dropout_fraction) or not 0 <= self.max_dropout_fraction < 1:
            raise ValueError("max_dropout_fraction must be in [0, 1).")
        if not math.isfinite(self.max_abs_update) or self.max_abs_update <= 0:
            raise ValueError("max_abs_update must be finite and positive.")
        if self.recovery_threshold is None:
            self.recovery_threshold = int(self.min_clients)
        if (
            not isinstance(self.recovery_threshold, Integral)
            or isinstance(self.recovery_threshold, bool)
            or self.recovery_threshold < 2
        ):
            raise ValueError("recovery_threshold must be an integer of at least 2.")
        if self.recovery_threshold > self.min_clients:
            raise ValueError("recovery_threshold cannot exceed min_clients.")


@dataclass(frozen=True)
class RecoveryShare:
    """One participant's Shamir share of a non-uploader's ephemeral key."""

    owner_id: int
    recipient_id: int
    round_id: str
    x: int
    y: int


@dataclass
class MaskedClientUpdate:
    """A client upload; values are modular masked integers, never plaintext."""

    client_id: int
    state_dict: Dict[str, torch.Tensor]
    num_samples: int
    round_id: str
    tensor_dtypes: Dict[str, torch.dtype]


def _validate_client_ids(client_ids: Iterable[int]) -> Tuple[int, ...]:
    ids = tuple(client_ids)
    if not ids or any(
        not isinstance(client_id, Integral) or isinstance(client_id, bool)
        for client_id in ids
    ):
        raise ValueError("client_ids must contain at least one integer client ID.")
    normalized = tuple(int(client_id) for client_id in ids)
    if len(set(normalized)) != len(normalized):
        raise ValueError("client_ids must be unique.")
    return tuple(sorted(normalized))


def _private_bytes(private_key: X25519PrivateKey) -> bytes:
    return private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _public_bytes(public_key: X25519PublicKey) -> bytes:
    return public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _split_secret(secret: bytes, shares_count: int, threshold: int) -> Sequence[Tuple[int, int]]:
    secret_value = int.from_bytes(secret, byteorder="big")
    coefficients = [secret_value] + [
        secrets.randbelow(_SHAMIR_FIELD) for _ in range(threshold - 1)
    ]
    shares = []
    for x in range(1, shares_count + 1):
        y = 0
        for coefficient in reversed(coefficients):
            y = (y * x + coefficient) % _SHAMIR_FIELD
        shares.append((x, y))
    return shares


def _reconstruct_secret(shares: Sequence[Tuple[int, int]], threshold: int) -> bytes:
    if len(shares) < threshold:
        raise ValueError(
            f"Dropout recovery requires at least {threshold} valid recovery shares."
        )
    selected = shares[:threshold]
    xs = [x for x, _ in selected]
    if len(set(xs)) != len(xs):
        raise ValueError("Recovery shares must have distinct x coordinates.")
    secret_value = 0
    for index, (x_i, y_i) in enumerate(selected):
        numerator = 1
        denominator = 1
        for other_index, (x_j, _) in enumerate(selected):
            if other_index == index:
                continue
            numerator = (numerator * -x_j) % _SHAMIR_FIELD
            denominator = (denominator * (x_i - x_j)) % _SHAMIR_FIELD
        lagrange = numerator * pow(denominator, -1, _SHAMIR_FIELD)
        secret_value = (secret_value + y_i * lagrange) % _SHAMIR_FIELD
    if secret_value >= 1 << 256:
        raise ValueError("Recovered private key is outside the X25519 key range.")
    return secret_value.to_bytes(32, byteorder="big")


def _derive_mask_key(shared_secret: bytes, round_id: str, left_id: int, right_id: int) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=hashlib.sha256(round_id.encode("utf-8")).digest(),
        info=f"flora-secagg-v1:{left_id}:{right_id}".encode("ascii"),
    ).derive(shared_secret)


def _expand_mask(
    key: bytes,
    tensor_name: str,
    shape: torch.Size,
    modulus: int,
) -> torch.Tensor:
    count = math.prod(shape)
    if count == 0:
        return torch.empty(shape, dtype=torch.int64)

    shape_context = ",".join(str(size) for size in shape)
    context = f"flora-secagg-mask-v1:{tensor_name}:{shape_context}:".encode("utf-8")
    values = np.empty(count, dtype=np.int64)
    bits = (modulus - 1).bit_length()
    candidate_mask = np.uint64((1 << bits) - 1)
    filled = 0
    counter = 0
    while filled < count:
        digest = hmac.new(
            key, context + counter.to_bytes(8, "big"), hashlib.sha256
        ).digest()
        candidates = np.frombuffer(digest, dtype="<u8").copy()
        candidates &= candidate_mask
        accepted = candidates[candidates < modulus]
        take = min(len(accepted), count - filled)
        if take:
            values[filled : filled + take] = accepted[:take].astype(np.int64)
            filled += take
        counter += 1
    return torch.from_numpy(values.astype(np.int64).reshape(tuple(shape)))


def _quantize_tensor(tensor: torch.Tensor, config: SecureAggregationConfig) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
        raise ValueError("Secure aggregation supports only dense tensor updates.")
    if tensor.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported tensor dtype for secure aggregation: {tensor.dtype}.")
    if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
        raise ValueError("Secure aggregation updates must contain only finite values.")
    value = tensor.detach().to(device="cpu", dtype=torch.float64)
    if value.numel() and value.abs().max().item() > config.max_abs_update:
        raise ValueError(
            f"Update magnitude exceeds configured max_abs_update={config.max_abs_update}."
        )
    scaled = torch.round(value * config.quantization_scale)
    if scaled.numel() and scaled.abs().max().item() > torch.iinfo(torch.int64).max:
        raise OverflowError("Fixed-point quantization exceeds the int64 range.")
    return scaled.to(dtype=torch.int64)


class SecureAggregationParticipant:
    """Client-side X25519 key, recovery shares, and update masking."""

    def __init__(self, client_id: int, config: SecureAggregationConfig):
        if not isinstance(client_id, Integral) or isinstance(client_id, bool):
            raise ValueError("client_id must be an integer.")
        self.client_id = int(client_id)
        self.config = config
        self._round_id: str | None = None
        self._last_round_id: str | None = None
        self._roster: Tuple[int, ...] = ()
        self._private_key: X25519PrivateKey | None = None
        self._public_keys: Dict[int, bytes] = {}
        self._recovery_shares: Dict[Tuple[str, int], RecoveryShare] = {}

    def begin_round(self, round_id: str, client_ids: Iterable[int]) -> bytes:
        if not isinstance(round_id, str) or not round_id:
            raise ValueError("round_id must be a non-empty string.")
        if round_id == self._last_round_id:
            raise ValueError("A secure aggregation round ID cannot be reused.")
        roster = _validate_client_ids(client_ids)
        if self.client_id not in roster:
            raise ValueError(f"Client {self.client_id} is not in this round's roster.")
        self._private_key = X25519PrivateKey.generate()
        self._round_id = round_id
        self._last_round_id = round_id
        self._roster = roster
        self._public_keys = {}
        self._recovery_shares.clear()
        return _public_bytes(self._private_key.public_key())

    def accept_public_keys(self, round_id: str, public_keys: Mapping[int, bytes]) -> None:
        if round_id != self._round_id or self._private_key is None:
            raise ValueError("Public keys do not belong to this participant's active round.")
        normalized = {int(client_id): key for client_id, key in public_keys.items()}
        if set(normalized) != set(self._roster):
            raise ValueError("Public-key roster does not match the round participant roster.")
        if any(not isinstance(key, bytes) or len(key) != 32 for key in normalized.values()):
            raise ValueError("Every X25519 public key must be exactly 32 bytes.")
        if normalized[self.client_id] != _public_bytes(self._private_key.public_key()):
            raise ValueError("The participant's public key does not match its private key.")
        self._public_keys = normalized

    def distribute_recovery_shares(
        self,
        recipients: Mapping[int, object],
        threshold: int,
    ) -> None:
        if self._private_key is None or not self._public_keys:
            raise ValueError("The participant must complete round key exchange first.")
        if set(recipients) != set(self._roster):
            raise ValueError("Recovery-share recipients must match the round roster.")
        if threshold < 2 or threshold > len(self._roster):
            raise ValueError("Recovery threshold must be between 2 and the roster size.")

        shares = _split_secret(_private_bytes(self._private_key), len(self._roster), threshold)
        for index, recipient_id in enumerate(self._roster):
            recipient = recipients[recipient_id]
            share = RecoveryShare(
                owner_id=self.client_id,
                recipient_id=recipient_id,
                round_id=self._round_id,
                x=shares[index][0],
                y=shares[index][1],
            )
            recipient.receive_secure_recovery_share(share)

    def receive_secure_recovery_share(self, share: RecoveryShare) -> None:
        if not isinstance(share, RecoveryShare):
            raise ValueError("Malformed secure aggregation recovery share.")
        if share.round_id != self._round_id or share.recipient_id != self.client_id:
            raise ValueError("Recovery share has an invalid round or recipient.")
        if share.owner_id not in self._roster or share.x <= 0 or not 0 <= share.y < _SHAMIR_FIELD:
            raise ValueError("Recovery share is outside the expected range.")
        key = (share.round_id, share.owner_id)
        if key in self._recovery_shares:
            raise ValueError("Duplicate recovery share.")
        self._recovery_shares[key] = share

    def recovery_share_for(self, owner_id: int, round_id: str) -> RecoveryShare:
        try:
            return self._recovery_shares[(round_id, int(owner_id))]
        except KeyError as exc:
            raise ValueError("No recovery share is available for that client and round.") from exc

    def finish_round(self, round_id: str) -> None:
        if round_id != self._round_id:
            raise ValueError("Cannot clear secure state for a different round.")
        self._private_key = None
        self._public_keys.clear()
        self._recovery_shares.clear()
        self._round_id = None

    def pairwise_mask(
        self,
        peer_id: int,
        tensor_name: str,
        shape: torch.Size,
    ) -> torch.Tensor:
        if self._private_key is None or not self._public_keys:
            raise ValueError("The participant must complete round key exchange first.")
        peer_id = int(peer_id)
        if peer_id == self.client_id or peer_id not in self._public_keys:
            raise ValueError("Pairwise masks require a different peer in the round roster.")
        lower_id, higher_id = sorted((self.client_id, peer_id))
        shared_secret = self._private_key.exchange(
            X25519PublicKey.from_public_bytes(self._public_keys[peer_id])
        )
        key = _derive_mask_key(shared_secret, self._round_id, lower_id, higher_id)
        mask = _expand_mask(key, tensor_name, shape, self.config.modulus)
        if self.client_id > peer_id:
            mask = torch.remainder(-mask, self.config.modulus)
        return mask

    def mask_client_update(
        self,
        state_dict: Mapping[str, torch.Tensor],
        num_samples: int,
    ) -> MaskedClientUpdate:
        if self._private_key is None or not self._public_keys:
            raise ValueError("The participant must complete round key exchange first.")
        if not isinstance(num_samples, Integral) or isinstance(num_samples, bool) or num_samples <= 0:
            raise ValueError("num_samples must be a positive integer.")
        if not isinstance(state_dict, Mapping) or not state_dict:
            raise ValueError("Client update must be a non-empty tensor mapping.")

        masked_state: Dict[str, torch.Tensor] = {}
        tensor_dtypes: Dict[str, torch.dtype] = {}
        for key, tensor in state_dict.items():
            if not isinstance(key, str):
                raise ValueError("Model state keys must be strings.")
            quantized = _quantize_tensor(tensor, self.config)
            if quantized.numel():
                largest = int(quantized.abs().max().item())
                if largest > torch.iinfo(torch.int64).max // int(num_samples):
                    raise OverflowError("Sample-weighted fixed-point update exceeds int64.")
            weighted = quantized * int(num_samples)
            masked = torch.remainder(weighted, self.config.modulus)
            for peer_id in self._roster:
                if peer_id != self.client_id:
                    masked = torch.remainder(
                        masked + self.pairwise_mask(peer_id, key, quantized.shape),
                        self.config.modulus,
                    )
            masked_state[key] = masked.to(dtype=torch.int64)
            tensor_dtypes[key] = tensor.dtype

        return MaskedClientUpdate(
            client_id=self.client_id,
            state_dict=masked_state,
            num_samples=int(num_samples),
            round_id=self._round_id,
            tensor_dtypes=tensor_dtypes,
        )


class SecureAggregationProtocol:
    """Server-side masked-sum aggregation and dropout-mask recovery."""

    def __init__(
        self,
        config: SecureAggregationConfig | None = None,
        client_ids: Iterable[int] | None = None,
    ):
        self.config = config or SecureAggregationConfig()
        self.client_ids = _validate_client_ids(client_ids) if client_ids is not None else ()
        self._round_id: str | None = None
        self._public_keys: Dict[int, bytes] = {}
        self._used_round_ids: Set[str] = set()

    def begin_round(self, round_id: str, public_keys: Mapping[int, bytes]) -> None:
        if not isinstance(round_id, str) or not round_id:
            raise ValueError("round_id must be a non-empty string.")
        if round_id in self._used_round_ids:
            raise ValueError("A secure aggregation round ID cannot be reused.")
        roster = _validate_client_ids(public_keys)
        if self.client_ids and roster != self.client_ids:
            raise ValueError("Public-key roster does not match configured client IDs.")
        if len(roster) < self.config.min_clients:
            raise ValueError("Secure aggregation roster is below min_clients.")
        if self.config.recovery_threshold > len(roster):
            raise ValueError("recovery_threshold cannot exceed the round roster size.")
        if any(not isinstance(key, bytes) or len(key) != 32 for key in public_keys.values()):
            raise ValueError("Every X25519 public key must be exactly 32 bytes.")
        self.client_ids = roster
        self._round_id = round_id
        self._public_keys = {int(client_id): key for client_id, key in public_keys.items()}

    def quantize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return signed fixed-point integers used by client masking."""
        return _quantize_tensor(tensor, self.config)

    def dequantize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Decode a signed fixed-point tensor without modular interpretation."""
        if not isinstance(tensor, torch.Tensor) or not torch.isfinite(
            tensor.to(dtype=torch.float64)
        ).all().item():
            raise ValueError("dequantize_tensor requires a finite tensor.")
        return tensor.to(dtype=torch.float64) / self.config.quantization_scale

    def _recover_dropout_mask(
        self,
        dropped_id: int,
        submitted_ids: Set[int],
        shares: Sequence[RecoveryShare],
        key_shapes: Mapping[str, torch.Size],
    ) -> Dict[str, torch.Tensor]:
        if not self.config.allow_recovery:
            raise ValueError("A client dropped before upload and recovery is disabled.")

        valid_shares = []
        seen_recipients = set()
        for share in shares:
            if not isinstance(share, RecoveryShare):
                raise ValueError("Malformed dropout recovery share.")
            if (
                share.owner_id != dropped_id
                or share.round_id != self._round_id
                or share.recipient_id not in submitted_ids
                or share.recipient_id in seen_recipients
                or not 0 < share.x < _SHAMIR_FIELD
                or not 0 <= share.y < _SHAMIR_FIELD
            ):
                raise ValueError("Invalid or duplicate dropout recovery share.")
            seen_recipients.add(share.recipient_id)
            valid_shares.append((share.x, share.y))

        private_bytes = _reconstruct_secret(
            valid_shares, int(self.config.recovery_threshold)
        )
        private_key = X25519PrivateKey.from_private_bytes(private_bytes)
        if _public_bytes(private_key.public_key()) != self._public_keys[dropped_id]:
            raise ValueError("Recovery shares do not match the dropped client's public key.")

        residual_masks = {
            key: torch.zeros(shape, dtype=torch.int64) for key, shape in key_shapes.items()
        }
        for uploader_id in submitted_ids:
            lower_id, higher_id = sorted((dropped_id, uploader_id))
            shared_secret = private_key.exchange(
                X25519PublicKey.from_public_bytes(self._public_keys[uploader_id])
            )
            mask_key = _derive_mask_key(
                shared_secret, self._round_id, lower_id, higher_id
            )
            for tensor_name, shape in key_shapes.items():
                mask = _expand_mask(
                    mask_key, tensor_name, shape, self.config.modulus
                )
                if uploader_id < dropped_id:
                    residual_masks[tensor_name] = torch.remainder(
                        residual_masks[tensor_name] + mask, self.config.modulus
                    )
                else:
                    residual_masks[tensor_name] = torch.remainder(
                        residual_masks[tensor_name] - mask, self.config.modulus
                    )
        return residual_masks

    def aggregate(
        self,
        masked_updates: Sequence[MaskedClientUpdate],
        active_client_ids: Iterable[int] | None = None,
        sample_counts: Mapping[int, int] | None = None,
        recovery_shares: Mapping[int, Sequence[RecoveryShare]] | None = None,
    ) -> Dict[str, torch.Tensor]:
        if self._round_id is None:
            raise ValueError("begin_round must be called before aggregation.")
        if self._round_id in self._used_round_ids:
            raise ValueError("This secure aggregation round has already been completed.")
        if not masked_updates:
            raise ValueError("No masked client updates to aggregate.")

        roster = set(self.client_ids if active_client_ids is None else _validate_client_ids(active_client_ids))
        if roster != set(self.client_ids):
            raise ValueError("Aggregation roster must match the round's configured participants.")

        seen_clients: Set[int] = set()
        for update in masked_updates:
            if not isinstance(update, MaskedClientUpdate):
                raise ValueError("Malformed secure aggregation submission.")
            if (
                not isinstance(update.client_id, Integral)
                or isinstance(update.client_id, bool)
            ):
                raise ValueError("Malformed secure aggregation client ID.")
            if update.client_id in seen_clients:
                raise ValueError(f"Duplicate client submission for id {update.client_id}.")
            seen_clients.add(update.client_id)
            if update.client_id not in roster:
                raise ValueError(f"Client {update.client_id} is not an active participant.")

        if len(seen_clients) < self.config.min_clients:
            raise ValueError(
                f"Secure aggregation requires at least {self.config.min_clients} uploading clients."
            )
        missing_ids = roster - seen_clients
        dropout_fraction = len(missing_ids) / len(roster)
        if dropout_fraction > self.config.max_dropout_fraction:
            raise ValueError(
                "Secure aggregation exceeded the configured dropout budget: "
                f"{dropout_fraction:.2%} > {self.config.max_dropout_fraction:.2%}."
            )
        if missing_ids and not self.config.allow_recovery:
            raise ValueError("A client dropped before upload and recovery is disabled.")

        updates_by_id = {update.client_id: update for update in masked_updates}
        if sample_counts is not None and set(sample_counts) != seen_clients:
            raise ValueError("sample_counts must contain exactly the uploading client IDs.")
        counts: Dict[int, int] = {}
        for client_id, update in updates_by_id.items():
            count = update.num_samples if sample_counts is None else sample_counts[client_id]
            if (
                not isinstance(count, Integral)
                or isinstance(count, bool)
                or count <= 0
                or int(count) != update.num_samples
            ):
                raise ValueError(f"Client {client_id} submitted an invalid sample count.")
            counts[client_id] = int(count)
            if update.round_id != self._round_id:
                raise ValueError(f"Client {client_id} submitted an update for the wrong round.")
            if (
                not isinstance(update.state_dict, Mapping)
                or not update.state_dict
                or not isinstance(update.tensor_dtypes, Mapping)
                or set(update.state_dict) != set(update.tensor_dtypes)
            ):
                raise ValueError(f"Client {client_id} submitted a malformed state dictionary.")

        total_samples = sum(counts.values())
        conservative_bound = (
            total_samples
            * (self.config.max_abs_update * self.config.quantization_scale + 0.5)
        )
        if not math.isfinite(conservative_bound) or conservative_bound >= self.config.modulus // 2:
            raise OverflowError(
                "Configured update bound and sample counts could wrap the signed modular sum."
            )

        first_update = masked_updates[0]
        reference_keys = set(first_update.state_dict)
        reference_shapes: Dict[str, torch.Size] = {}
        reference_dtypes = first_update.tensor_dtypes
        for key, tensor in first_update.state_dict.items():
            if (
                not isinstance(key, str)
                or not isinstance(tensor, torch.Tensor)
                or tensor.dtype != torch.int64
                or tensor.layout != torch.strided
            ):
                raise ValueError(f"Malformed masked tensor for '{key}'.")
            tensor = tensor.detach().to(device="cpu")
            if torch.any(tensor < 0).item() or torch.any(tensor >= self.config.modulus).item():
                raise ValueError(f"Masked tensor '{key}' is outside the modular field.")
            if reference_dtypes[key] not in _SUPPORTED_DTYPES:
                raise ValueError(f"Unsupported original tensor dtype for '{key}'.")
            reference_shapes[key] = tensor.shape

        accumulated = {
            key: torch.zeros(shape, dtype=torch.int64) for key, shape in reference_shapes.items()
        }
        for update in masked_updates:
            if set(update.state_dict) != reference_keys:
                raise ValueError(
                    f"Client {update.client_id} submitted incompatible model keys."
                )
            if update.tensor_dtypes != reference_dtypes:
                raise ValueError(
                    f"Client {update.client_id} submitted incompatible tensor dtypes."
                )
            for key, tensor in update.state_dict.items():
                if (
                    not isinstance(tensor, torch.Tensor)
                    or tensor.dtype != torch.int64
                    or tensor.layout != torch.strided
                    or tensor.shape != reference_shapes[key]
                ):
                    raise ValueError(
                        f"Client {update.client_id} submitted an incompatible masked tensor '{key}'."
                    )
                tensor = tensor.detach().to(device="cpu")
                if torch.any(tensor < 0).item() or torch.any(tensor >= self.config.modulus).item():
                    raise ValueError(
                        f"Client {update.client_id} submitted out-of-range values for '{key}'."
                    )
                accumulated[key] = torch.remainder(
                    accumulated[key] + tensor, self.config.modulus
                )

        if missing_ids:
            if recovery_shares is None or set(recovery_shares) != missing_ids:
                raise ValueError("Recovery shares are required for every client that did not upload.")
            for dropped_id in sorted(missing_ids):
                residual_masks = self._recover_dropout_mask(
                    dropped_id,
                    seen_clients,
                    recovery_shares[dropped_id],
                    reference_shapes,
                )
                for key in reference_keys:
                    accumulated[key] = torch.remainder(
                        accumulated[key] - residual_masks[key], self.config.modulus
                    )

        aggregate_state: Dict[str, torch.Tensor] = {}
        for key, modular_sum in accumulated.items():
            signed_sum = torch.where(
                modular_sum > self.config.modulus // 2,
                modular_sum - self.config.modulus,
                modular_sum,
            )
            average = signed_sum.to(torch.float64) / (
                self.config.quantization_scale * total_samples
            )
            original_dtype = reference_dtypes[key]
            if original_dtype.is_floating_point:
                if average.numel() and average.abs().max().item() > torch.finfo(original_dtype).max:
                    raise OverflowError(
                        f"Aggregated tensor '{key}' exceeds the representable range of {original_dtype}."
                    )
                aggregate_state[key] = average.to(dtype=original_dtype)
            else:
                rounded = average.round()
                integer_range = torch.iinfo(original_dtype)
                if rounded.numel() and (
                    rounded.min().item() < integer_range.min
                    or rounded.max().item() > integer_range.max
                ):
                    raise OverflowError(
                        f"Aggregated tensor '{key}' exceeds the representable range of {original_dtype}."
                    )
                aggregate_state[key] = rounded.to(dtype=original_dtype)

        self._used_round_ids.add(self._round_id)
        return aggregate_state

    def aggregate_weighted(
        self,
        masked_updates: Sequence[MaskedClientUpdate],
        active_client_ids: Iterable[int] | None = None,
        sample_counts: Mapping[int, int] | None = None,
        recovery_shares: Mapping[int, Sequence[RecoveryShare]] | None = None,
    ) -> Dict[str, torch.Tensor]:
        return self.aggregate(
            masked_updates,
            active_client_ids=active_client_ids,
            sample_counts=sample_counts,
            recovery_shares=recovery_shares,
        )

    def __repr__(self) -> str:
        return (
            f"SecureAggregationProtocol(client_ids={self.client_ids}, "
            f"round_active={self._round_id is not None})"
        )
