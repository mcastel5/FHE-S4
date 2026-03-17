import argparse
import time

import numpy as np
import tenseal as ts
import torch

from tenseal_context import make_ckks_context

try:
    from s4d.model import MiniS4D  # type: ignore
except Exception:
    from train_mini_s4d import MiniS4D

_TOEPLITZ_MAT_CACHE: dict[tuple[int, bytes], list[list[float]]] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference sanity shell: plain, FHE stub, and FHE Toeplitz milestones."
    )
    parser.add_argument(
        "--mode",
        choices=("plain", "fhe_stub", "fhe_toeplitz", "fhe_full", "both"),
        default="both",
    )
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--poly_modulus_degree", type=int, default=8192)
    parser.add_argument("--stub_bias", type=float, default=0.01)
    parser.add_argument(
        "--toeplitz_K",
        type=int,
        default=16,
        help="Number of kernel taps to use in Toeplitz op (<= seq_len).",
    )
    return parser.parse_args()


def pack_per_channel(x_np: np.ndarray) -> list[np.ndarray]:
    """Pack input by channel for CKKS.

    Args:
        x_np: shape (L, d_model)
    Returns:
        List of d_model vectors, each shape (L,).
    """
    if x_np.ndim != 2:
        raise ValueError(f"x_np must be rank-2, got shape {x_np.shape}")
    return [x_np[:, c].astype(np.float64, copy=False) for c in range(x_np.shape[1])]


def encrypt_per_channel(ctx: ts.Context, chans: list[np.ndarray]) -> list[ts.CKKSVector]:
    """Encrypt one ciphertext per channel.

    We keep one CKKSVector per channel to avoid cross-channel slot mixing when
    applying Toeplitz/conv logic based on rotations.
    """
    out: list[ts.CKKSVector] = []
    for arr in chans:
        out.append(ts.ckks_vector(ctx, arr.tolist()))
    return out


def toeplitz_plain(u: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Causal Toeplitz operator in plaintext.

    Convention:
      y[t] = sum_{k=0..K-1, k<=t} coeffs[k] * u[t-k]
    This matches the lower-triangular Toeplitz conv used by S4D kernels.
    """
    L = u.shape[0]
    T = _build_causal_toeplitz(coeffs, L)
    return u @ T


def _build_causal_toeplitz(coeffs: np.ndarray, L: int) -> np.ndarray:
    """Build an LxL causal Toeplitz matrix for row-vector matmul.

    TenSEAL's `ckks_vector.matmul(M)` follows row-vector semantics: `y = x @ M`.
    So we place coeffs on upper diagonals:
      M[i, i+k] = coeffs[k]
    to realize:
      y[t] = sum_{k>=0} coeffs[k] * x[t-k]
    """
    K = min(L, coeffs.shape[0])
    T = np.zeros((L, L), dtype=np.float64)
    for k in range(K):
        row_idx = np.arange(0, L - k)
        col_idx = row_idx + k
        T[row_idx, col_idx] = coeffs[k]
    return T


def toeplitz_fhe(u_enc: ts.CKKSVector, coeffs: np.ndarray) -> ts.CKKSVector:
    """Apply causal Toeplitz operator in encrypted space.

    Why Galois keys are needed:
    - Toeplitz/conv in packed CKKS is implemented with slot rotations.
    - Mathematically this is y = sum_k coeffs[k] * rotate(u, k).
    - TenSEAL's Python API does not expose explicit rotate for CKKSVector, so we
      use `matmul` with an equivalent Toeplitz matrix; internally this still uses
      rotations and therefore needs Galois keys.
    - Therefore the context must include Galois keys for this operation.
    """
    L = int(u_enc.size())
    key = (L, np.asarray(coeffs, dtype=np.float64).tobytes())
    T_list = _TOEPLITZ_MAT_CACHE.get(key)
    if T_list is None:
        T = _build_causal_toeplitz(coeffs, L)
        T_list = T.tolist()
        _TOEPLITZ_MAT_CACHE[key] = T_list
    return u_enc.matmul(T_list)


def extract_channel_kernel_coeffs(
    model: MiniS4D, seq_len: int, d_model: int, toeplitz_K: int
) -> list[np.ndarray]:
    """Extract per-channel Toeplitz coefficients without building giant matrices.

    Preferred path: model.kernel_gen(seq_len) -> shape (d_model, seq_len)
    Fallback path: model.export_toeplitz(head_idx) and extract first column once.
    """
    K = min(seq_len, toeplitz_K)

    with torch.no_grad():
        if hasattr(model, "kernel_gen"):
            kernel = model.kernel_gen(seq_len)
            if not isinstance(kernel, torch.Tensor):
                raise TypeError("model.kernel_gen(seq_len) did not return a torch.Tensor")
            k_np = kernel.detach().cpu().numpy().astype(np.float64)
            if k_np.shape != (d_model, seq_len):
                raise ValueError(f"Unexpected kernel shape {k_np.shape}")
            return [k_np[c, :K].copy() for c in range(d_model)]

    if hasattr(model, "export_toeplitz"):
        coeffs: list[np.ndarray] = []
        for c in range(d_model):
            T = np.asarray(model.export_toeplitz(head_idx=c), dtype=np.float64)
            if T.shape != (seq_len, seq_len):
                raise ValueError(f"Unexpected Toeplitz shape for channel {c}: {T.shape}")
            coeffs.append(T[:, 0].copy()[:K])
            del T
        return coeffs

    raise RuntimeError("Unable to extract Toeplitz/kernel coefficients from MiniS4D model.")


def fhe_forward_stub(x_enc: ts.CKKSVector, bias_flat: np.ndarray) -> ts.CKKSVector:
    """Milestone 1: tiny encrypted op (ciphertext + plaintext bias)."""
    return x_enc + bias_flat.tolist()


def main() -> None:
    args = parse_args()

    # -------------------------------------------------------------------------
    # Stage 0: static setup for inference-only sanity checks.
    # Guardrails:
    # - batch_size is fixed to 1
    # - CPU only
    # - no training in this script
    # -------------------------------------------------------------------------
    batch_size = 1
    device = torch.device("cpu")
    seq_len = args.seq_len
    d_model = args.d_model

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    x_np = np.random.randn(seq_len, d_model).astype(np.float64)
    x_torch = torch.from_numpy(x_np.astype(np.float32)).transpose(0, 1).unsqueeze(0).to(device)
    if x_torch.shape != (batch_size, d_model, seq_len):
        raise RuntimeError(f"unexpected x_torch shape: {tuple(x_torch.shape)}")
    print(f"[setup] x_np shape={x_np.shape}, x_torch shape={tuple(x_torch.shape)}")

    model = MiniS4D(d_model=d_model, L=seq_len, dropout=0.0).to(device)
    model.eval()

    # -------------------------------------------------------------------------
    # Stage 1: optional plain baseline.
    # Job: run the original MiniS4D forward once, report timing and basic stats.
    # -------------------------------------------------------------------------
    y_plain_model: np.ndarray | None = None
    if args.mode in ("plain", "both"):
        with torch.no_grad():
            t0 = time.perf_counter()
            y_plain = model(x_torch)
            t1 = time.perf_counter()
        y_plain_model = y_plain.detach().cpu().numpy()
        print(f"[plain] output_shape={tuple(y_plain_model.shape)}")
        print(f"[plain] forward_time_s={t1 - t0:.6f}")
        print(
            "[plain] output_stats="
            f"mean={float(y_plain_model.mean()):.6e}, "
            f"std={float(y_plain_model.std()):.6e}, "
            f"min={float(y_plain_model.min()):.6e}, "
            f"max={float(y_plain_model.max()):.6e}"
        )

    # -------------------------------------------------------------------------
    # Stage 2: one-time CKKS context creation (for FHE modes).
    # Critical guardrail:
    # - Context is created ONCE in main(), never inside forward, never in loops.
    # - Enable Galois keys when using Toeplitz because rotations are required.
    # -------------------------------------------------------------------------
    fhe_modes = ("fhe_stub", "fhe_toeplitz", "fhe_full", "both")
    use_fhe = args.mode in fhe_modes
    use_toeplitz = args.mode in ("fhe_toeplitz", "fhe_full")

    ctx: ts.Context | None = None
    if use_fhe:
        t_ctx0 = time.perf_counter()
        ctx = make_ckks_context(
            poly_modulus_degree=args.poly_modulus_degree,
            need_galois=use_toeplitz,
            need_relin=False,
        )
        t_ctx1 = time.perf_counter()
        print(f"[fhe] context_creation_time_s={t_ctx1 - t_ctx0:.6f}")
        print(f"[fhe] galois_keys_enabled={use_toeplitz}")

    # -------------------------------------------------------------------------
    # Stage 3a: FHE stub milestone (default in --mode both).
    # Job: cheap end-to-end encrypted pass on flattened vector.
    # -------------------------------------------------------------------------
    if args.mode in ("fhe_stub", "both"):
        assert ctx is not None
        flat_x = x_np.reshape(-1).astype(np.float64)
        bias_flat = np.full_like(flat_x, args.stub_bias)
        y_stub_plain_flat = flat_x + bias_flat

        t_e0 = time.perf_counter()
        x_enc = ts.ckks_vector(ctx, flat_x.tolist())
        t_e1 = time.perf_counter()

        t_o0 = time.perf_counter()
        y_enc = fhe_forward_stub(x_enc, bias_flat)
        t_o1 = time.perf_counter()

        t_d0 = time.perf_counter()
        y_dec = np.array(y_enc.decrypt(), dtype=np.float64)
        t_d1 = time.perf_counter()

        max_abs_diff = float(np.max(np.abs(y_dec - y_stub_plain_flat)))
        print(f"[fhe_stub] flat_len={flat_x.shape[0]}")
        print(f"[fhe_stub] encrypt_time_s={t_e1 - t_e0:.6f}")
        print(f"[fhe_stub] op_time_s={t_o1 - t_o0:.6f}")
        print(f"[fhe_stub] decrypt_time_s={t_d1 - t_d0:.6f}")
        print(f"[fhe_stub] max_abs_diff={max_abs_diff:.6e}")

    # -------------------------------------------------------------------------
    # Stage 3b: FHE Toeplitz milestone (primary objective).
    # Job:
    # - pack one CKKSVector per channel (length L each)
    # - apply per-channel Toeplitz op in encrypted domain
    # - decrypt and compare against plaintext Toeplitz reference
    # -------------------------------------------------------------------------
    if args.mode in ("fhe_toeplitz", "fhe_full"):
        assert ctx is not None
        K = min(seq_len, args.toeplitz_K)
        print(f"[fhe_toeplitz] using seq_len={seq_len}, d_model={d_model}, K={K}")

        t_coeff0 = time.perf_counter()
        coeffs_by_channel = extract_channel_kernel_coeffs(model, seq_len, d_model, K)
        t_coeff1 = time.perf_counter()
        print(f"[fhe_toeplitz] coeff_extract_time_s={t_coeff1 - t_coeff0:.6f}")
        print(f"[fhe_toeplitz] coeffs_shape=({len(coeffs_by_channel)}, {K})")

        chans_plain = pack_per_channel(x_np)
        print(f"[fhe_toeplitz] packed_channels={len(chans_plain)}, channel_len={seq_len}")

        # Encryption timings per channel + total
        enc_per_channel: list[float] = []
        enc_chans: list[ts.CKKSVector] = []
        t_enc_total0 = time.perf_counter()
        for c in range(d_model):
            t0 = time.perf_counter()
            enc_chans.append(ts.ckks_vector(ctx, chans_plain[c].tolist()))
            t1 = time.perf_counter()
            enc_per_channel.append(t1 - t0)
        t_enc_total1 = time.perf_counter()

        # Toeplitz op timings per channel + total
        op_per_channel: list[float] = []
        y_enc_chans: list[ts.CKKSVector] = []
        t_op_total0 = time.perf_counter()
        for c in range(d_model):
            t0 = time.perf_counter()
            y_enc_chans.append(toeplitz_fhe(enc_chans[c], coeffs_by_channel[c]))
            t1 = time.perf_counter()
            op_per_channel.append(t1 - t0)
        t_op_total1 = time.perf_counter()

        # Decryption timings per channel + total
        dec_per_channel: list[float] = []
        y_dec_chans: list[np.ndarray] = []
        t_dec_total0 = time.perf_counter()
        for c in range(d_model):
            t0 = time.perf_counter()
            y_dec_chans.append(np.array(y_enc_chans[c].decrypt(), dtype=np.float64))
            t1 = time.perf_counter()
            dec_per_channel.append(t1 - t0)
        t_dec_total1 = time.perf_counter()

        # Plaintext Toeplitz reference timings
        t_ref0 = time.perf_counter()
        y_plain_ref_chans = [
            toeplitz_plain(chans_plain[c], coeffs_by_channel[c]) for c in range(d_model)
        ]
        t_ref1 = time.perf_counter()

        # Error report per channel + overall
        per_channel_max = []
        for c in range(d_model):
            err_c = float(np.max(np.abs(y_dec_chans[c] - y_plain_ref_chans[c])))
            per_channel_max.append(err_c)
            print(f"[fhe_toeplitz] channel={c} max_abs_diff={err_c:.6e}")
        overall_max = float(np.max(per_channel_max))
        print(f"[fhe_toeplitz] overall_max_abs_diff={overall_max:.6e}")

        y_dec_matrix = np.stack(y_dec_chans, axis=1)  # (L, d_model)
        print(f"[fhe_toeplitz] decrypted_matrix_shape={y_dec_matrix.shape}")
        print(f"[fhe_toeplitz] encrypt_total_s={t_enc_total1 - t_enc_total0:.6f}")
        print(
            "[fhe_toeplitz] encrypt_per_channel_s="
            + ",".join(f"{v:.6f}" for v in enc_per_channel)
        )
        print(f"[fhe_toeplitz] toeplitz_total_s={t_op_total1 - t_op_total0:.6f}")
        print(
            "[fhe_toeplitz] toeplitz_per_channel_s="
            + ",".join(f"{v:.6f}" for v in op_per_channel)
        )
        print(f"[fhe_toeplitz] decrypt_total_s={t_dec_total1 - t_dec_total0:.6f}")
        print(
            "[fhe_toeplitz] decrypt_per_channel_s="
            + ",".join(f"{v:.6f}" for v in dec_per_channel)
        )
        print(f"[fhe_toeplitz] plain_reference_time_s={t_ref1 - t_ref0:.6f}")

        # ---------------------------------------------------------------------
        # Stage 3c (optional): hybrid "fhe_full" milestone.
        # Job:
        # - keep Toeplitz in FHE
        # - decrypt
        # - run remaining nonlinear/output parts in plaintext
        #
        # We intentionally do NOT attempt GLU/1x1 conv fully in FHE here.
        # ---------------------------------------------------------------------
        if args.mode == "fhe_full":
            with torch.no_grad():
                y_torch = (
                    torch.from_numpy(y_dec_matrix.astype(np.float32))
                    .transpose(0, 1)
                    .unsqueeze(0)
                    .to(device)
                )  # (1, d_model, L)

                # Continue from MiniS4D post-conv path in plaintext.
                y_hybrid = y_torch + x_torch * model.D.unsqueeze(-1)
                y_hybrid = model.dropout(model.activation(y_hybrid))
                y_hybrid = model.output_linear(y_hybrid)
                y_hybrid_out = model.decoder(y_hybrid.mean(dim=-1))
                y_hybrid_np = y_hybrid_out.detach().cpu().numpy()

            print(f"[fhe_full] hybrid_output_shape={tuple(y_hybrid_np.shape)}")
            if y_plain_model is not None:
                diff = float(np.max(np.abs(y_hybrid_np - y_plain_model)))
                print(f"[fhe_full] max_abs_diff_vs_plain_model={diff:.6e}")
            else:
                print("[fhe_full] plain baseline not run in this mode.")


if __name__ == "__main__":
    main()
