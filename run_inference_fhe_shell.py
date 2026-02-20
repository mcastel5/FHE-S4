import argparse
import sys
import time
from pathlib import Path

import numpy as np
import tenseal as ts
import torch
from tenseal_context import make_ckks_context

# Allow running as: python scripts/run_inference_fhe_shell.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


try:
    # Common package-style layout.
    from s4d.model import MiniS4D  # type: ignore
except Exception:
    # Fallback for this repository layout.
    from train_mini_s4d import MiniS4D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plain + minimal FHE inference shell.")
    parser.add_argument("--mode", choices=("plain", "fhe", "both"), default="both")
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--poly_modulus_degree", type=int, default=8192)
    return parser.parse_args()


def fhe_forward_stub(
    x_enc: ts.CKKSVector, seq_len: int, d_model: int, params: dict
) -> ts.CKKSVector:
    """Minimal encrypted forward stub.

    TODO: Replace bias-add stub with encrypted linear operations from S4D (Toeplitz/conv).
    TODO: Cache plaintext constants outside the per-call path.
    """
    expected = seq_len * d_model
    bias_flat = np.asarray(params["bias_flat"], dtype=np.float64)
    if bias_flat.shape != (expected,):
        raise ValueError(f"bias_flat must have shape ({expected},), got {bias_flat.shape}")
    return x_enc + bias_flat.tolist()


def main() -> None:
    args = parse_args()

    # Guardrail: start with batch_size=1 and small sizes.
    batch_size = 1
    device = torch.device("cpu")

    # Deterministic inputs and model init.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    seq_len = args.seq_len
    d_model = args.d_model

    # x_np shape is (seq_len, d_model); flattened for CKKS path later.
    x_np = np.random.randn(seq_len, d_model).astype(np.float64)
    x_torch = torch.from_numpy(x_np.astype(np.float32)).transpose(0, 1).unsqueeze(0).to(device)
    if x_torch.shape != (batch_size, d_model, seq_len):
        raise RuntimeError(f"Unexpected input shape {tuple(x_torch.shape)}")

    model = MiniS4D(d_model=d_model, L=seq_len, dropout=0.0).to(device)
    model.eval()

    # Build context ONCE at start if mode includes FHE.
    # Guardrail: do not create context in forward/per-call paths.
    # Guardrail: do not call generate_galois_keys unless rotations are needed.
    ctx = None
    if args.mode in ("fhe", "both"):
        t_ctx0 = time.perf_counter()
        ctx = make_ckks_context(
            poly_modulus_degree=args.poly_modulus_degree,
            need_galois=True,
            need_relin=False,
        )
        t_ctx1 = time.perf_counter()
        print(f"[fhe] context_creation_time_s: {t_ctx1 - t_ctx0:.6f}")

    if args.mode in ("plain", "both"):
        with torch.no_grad():
            t0 = time.perf_counter()
            y_plain_model = model(x_torch)
            t1 = time.perf_counter()
        y_plain_np = y_plain_model.detach().cpu().numpy()
        print(f"[plain] input_shape: {tuple(x_torch.shape)}")
        print(f"[plain] output_shape: {tuple(y_plain_np.shape)}")
        print(f"[plain] forward_time_s: {t1 - t0:.6f}")
        print(
            "[plain] output_stats: "
            f"mean={float(y_plain_np.mean()):.6e}, std={float(y_plain_np.std()):.6e}, "
            f"min={float(y_plain_np.min()):.6e}, max={float(y_plain_np.max()):.6e}"
        )

    if args.mode in ("fhe", "both"):
        assert ctx is not None
        flat_x = x_np.T.reshape(-1).astype(np.float64)
        bias_flat = np.full(seq_len * d_model, 0.01, dtype=np.float64)
        params = {"bias_flat": bias_flat}

        # Plaintext equivalent for stub validation.
        y_stub_plain_flat = flat_x + bias_flat

        t2 = time.perf_counter()
        x_enc = ts.ckks_vector(ctx, flat_x.tolist())
        t3 = time.perf_counter()

        t4 = time.perf_counter()
        y_enc = model(x_enc, context=ctx)
        t5 = time.perf_counter()

        t6 = time.perf_counter()
        y_dec = np.array(y_enc.decrypt(), dtype=np.float64)
        t7 = time.perf_counter()

        max_abs_diff = float(np.max(np.abs(y_dec - y_stub_plain_flat)))

        print(f"[fhe] flat_input_len: {flat_x.shape[0]}")
        print(f"[fhe] decrypted_output_shape: {tuple(y_dec.shape)}")
        print(f"[fhe] encrypt_time_s: {t3 - t2:.6f}")
        print(f"[fhe] fhe_op_time_s: {t5 - t4:.6f}")
        print(f"[fhe] decrypt_time_s: {t7 - t6:.6f}")
        print(f"[fhe] max_abs_diff_vs_plain_stub: {max_abs_diff:.6e}")


if __name__ == "__main__":
    main()