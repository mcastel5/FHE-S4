"""Utilities for creating and persisting a TenSEAL CKKS context.

Keep context creation lightweight and perform it ONCE at program start.
Do not create a TenSEAL context inside training or inference loops.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import tenseal as ts


def make_ckks_context(
    poly_modulus_degree: int = 8192,
    coeff_mod_bit_sizes: Sequence[int] = (60, 40, 40, 60),
    global_scale: float = 2**40,
    need_galois: bool = False,
    need_relin: bool = False,
) -> ts.Context:
    """Create a CKKS context with safe defaults for small inference demos.

    Important:
    - Create this context once and reuse it.
    - Do not build contexts in training/inference loops.
    - Galois/relinearization keys are generated only when explicitly requested.
    """

    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=list(coeff_mod_bit_sizes),
    )
    ctx.global_scale = global_scale

    if need_galois:
        ctx.generate_galois_keys()
    if need_relin:
        ctx.generate_relin_keys()

    return ctx


def serialize_context(ctx: ts.Context, path: str | Path, include_secret_key: bool = False) -> None:
    """Serialize a TenSEAL context to disk.

    Args:
        ctx: TenSEAL context object.
        path: Output file path.
        include_secret_key: Include secret key when True. Keep False for sharing.
    """

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = ctx.serialize(save_secret_key=include_secret_key)
    output_path.write_bytes(data)


def load_context(path: str | Path) -> ts.Context:
    """Load a TenSEAL context from disk."""

    data = Path(path).read_bytes()
    return ts.context_from(data)


def _bench_encrypt_decrypt() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(256).astype(np.float64)

    t0 = time.perf_counter()
    ctx = make_ckks_context()
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    x_enc = ts.ckks_vector(ctx, x.tolist())
    t3 = time.perf_counter()

    t4 = time.perf_counter()
    x_dec = np.array(x_enc.decrypt(), dtype=np.float64)
    t5 = time.perf_counter()

    max_abs_err = float(np.max(np.abs(x - x_dec)))

    print(f"context_creation_time_s: {t1 - t0:.6f}")
    print(f"encryption_time_s: {t3 - t2:.6f}")
    print(f"decryption_time_s: {t5 - t4:.6f}")
    print(f"max_abs_error: {max_abs_err:.6e}")


if __name__ == "__main__":
    _bench_encrypt_decrypt()