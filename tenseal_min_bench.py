import time
import numpy as np
import tenseal as ts


def main() -> None:
    np.random.seed(0)

    # CKKS context (no Galois keys generated on purpose for this benchmark)
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    context.global_scale = 2**40

    # Random input vector
    x = np.random.randn(256).astype(np.float64)

    # Encrypt
    t0 = time.perf_counter()
    x_enc = ts.ckks_vector(context, x.tolist())
    t1 = time.perf_counter()
    enc_time = t1 - t0

    # Decrypt
    t2 = time.perf_counter()
    x_dec = np.array(x_enc.decrypt(), dtype=np.float64)
    t3 = time.perf_counter()
    dec_time = t3 - t2

    x_err = np.max(np.abs(x - x_dec))

    print(f"encryption_time_s: {enc_time:.6f}")
    print(f"decryption_time_s: {dec_time:.6f}")
    print(f"x_max_abs_error: {x_err:.6e}")

    # Plaintext linear algebra target
    W = np.random.randn(256, 256).astype(np.float64)
    y_plain_full = x @ W

    # Try full encrypted plaintext matmul first.
    # If unavailable without Galois keys, fall back to a smaller linear op:
    # diagonal linear map y = x * diag(W), which is supported via element-wise multiply.
    try:
        t4 = time.perf_counter()
        y_enc = x_enc.matmul(W.tolist())
        t5 = time.perf_counter()
        y_dec = np.array(y_enc.decrypt(), dtype=np.float64)
        y_err = np.max(np.abs(y_plain_full - y_dec))
        print(f"linear_op: full_matmul_xW")
        print(f"linear_op_time_s: {t5 - t4:.6f}")
        print(f"y_max_abs_error_vs_y_plain: {y_err:.6e}")
    except Exception as exc:
        diag_w = np.diag(W).astype(np.float64)
        y_plain = x * diag_w
        t4 = time.perf_counter()
        y_enc = x_enc * diag_w.tolist()
        t5 = time.perf_counter()
        y_dec = np.array(y_enc.decrypt(), dtype=np.float64)
        y_err = np.max(np.abs(y_plain - y_dec))
        print(
            "linear_op: fallback_diagonal_map "
            "(full matmul likely requires rotations/Galois keys; skipped by design)"
        )
        print(f"linear_op_error_detail: {type(exc).__name__}: {exc}")
        print(f"linear_op_time_s: {t5 - t4:.6f}")
        print(f"y_max_abs_error_vs_y_plain: {y_err:.6e}")


if __name__ == "__main__":
    main()
