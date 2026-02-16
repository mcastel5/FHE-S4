import time
from typing import Optional

import numpy as np
import tenseal as ts


class FHELinear:
    def __init__(self, context: ts.Context, W: np.ndarray, b: Optional[np.ndarray] = None):
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError("W must be a square matrix of shape (d, d).")
        self.context = context
        self.W = np.asarray(W, dtype=np.float64)
        self.b = None if b is None else np.asarray(b, dtype=np.float64)
        if self.b is not None and self.b.shape != (self.W.shape[0],):
            raise ValueError("b must have shape (d,).")
        self._W_list = self.W.tolist()
        self._b_list = None if self.b is None else self.b.tolist()

    def __call__(self, x_enc: ts.CKKSVector) -> ts.CKKSVector:
        # For CKKS vectors, TenSEAL's plaintext matrix multiplication path is the
        # simplest full linear transform. It relies on rotations (Galois keys).
        y_enc = x_enc.matmul(self._W_list)
        if self._b_list is not None:
            y_enc = y_enc + self._b_list
        return y_enc


def main() -> None:
    np.random.seed(0)
    d = 128

    x = np.random.randn(d).astype(np.float64)
    W = np.random.randn(d, d).astype(np.float64)
    b = np.random.randn(d).astype(np.float64)
    y_plain = x @ W + b

    t0 = time.perf_counter()
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    context.global_scale = 2**40
    context.generate_galois_keys()
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    x_enc = ts.ckks_vector(context, x.tolist())
    t3 = time.perf_counter()

    fhe_linear = FHELinear(context, W, b)
    t4 = time.perf_counter()
    y_enc = fhe_linear(x_enc)
    t5 = time.perf_counter()

    t6 = time.perf_counter()
    y_dec = np.array(y_enc.decrypt(), dtype=np.float64)
    t7 = time.perf_counter()

    max_abs_error = np.max(np.abs(y_plain - y_dec))

    print(f"context_creation_time_s: {t1 - t0:.6f}")
    print(f"encryption_time_s: {t3 - t2:.6f}")
    print(f"encrypted_compute_time_s: {t5 - t4:.6f}")
    print(f"decryption_time_s: {t7 - t6:.6f}")
    print(f"max_abs_error: {max_abs_error:.6e}")


if __name__ == "__main__":
    main()
