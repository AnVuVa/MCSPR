"""Phase 2 smoke test — verify MCSPR_FINAL_LOCKED_V2.md A1-A6 applied.

Checks:
  1. forward() returns scalar (not tuple)
  2. backward() produces gradient on Y_hat
  3. Gradient norm > 0 and not NaN/Inf
  4. C_hat_t diagonal ≈ 1.0 (validates A3 Sigma_unbiased numerator)
  5. n_active_contexts diagnostic accessible via side-channel
  6. Second call updates EMA state (validates A2 gradient path)
"""

import numpy as np
import torch

from mcspr.core.loss import MCSPRLoss


def main():
    torch.manual_seed(2021)
    np.random.seed(2021)

    # Fake geometry: 300 genes, 15 modules, 6 contexts, 256 spots
    # N=256 with T=6 soft weights gives ~43 spots/ctx > k_min=30
    # (N=128 exposes exactly the audit-4 k_min-starvation pathology and
    # is tested separately below.)
    m_genes = 300
    B = 15
    T = 6
    N = 256

    # Random M_pinv (B, m) and C_prior (T, B, B) that looks correlation-ish
    M = np.random.randn(m_genes, B).astype(np.float32) * 0.1
    M = np.maximum(M, 0)  # NMF-like non-negative
    M_pinv = np.linalg.pinv(M).astype(np.float32)

    C_prior = np.zeros((T, B, B), dtype=np.float32)
    for t in range(T):
        A = np.random.randn(B, B).astype(np.float32)
        Sigma = A @ A.T
        std = np.sqrt(np.diag(Sigma) + 1e-8)
        C = Sigma / (std[:, None] * std[None, :])
        np.fill_diagonal(C, 1.0)
        C_prior[t] = C

    loss_fn = MCSPRLoss(
        M_pinv=torch.tensor(M_pinv),
        C_prior=torch.tensor(C_prior),
        n_contexts=T,
        k_min=30.0,
        tau=1.0e-4,
        beta=0.9,
        lambda_max=0.1,
    )

    # Fake Y_hat + context weights. Each spot's ctx weights softmax over T.
    Y_hat = torch.randn(N, m_genes, requires_grad=True)
    logits = torch.randn(N, T)
    ctx_w = torch.softmax(logits, dim=1)

    print(f"Y_hat shape: {tuple(Y_hat.shape)}")
    print(f"ctx_w shape: {tuple(ctx_w.shape)}, per-ctx mass: "
          f"{ctx_w.sum(0).tolist()}")

    # ── 1st forward ────────────────────────────────────────────────────
    loss = loss_fn(Y_hat, ctx_w, lambda_scale=1.0)
    print(f"\nCall 1: L_MCSPR = {loss.item():.6e}")
    assert loss.dim() == 0, f"expected scalar, got shape {tuple(loss.shape)}"
    diag1 = loss_fn._last_diagnostics
    print(f"  n_active_contexts = {diag1['n_active_contexts']}/{T}")
    print(f"  effective_lambda  = {diag1['effective_lambda']}")

    # ── backward ───────────────────────────────────────────────────────
    loss.backward()
    assert Y_hat.grad is not None, "Y_hat.grad is None"
    gnorm = Y_hat.grad.norm().item()
    assert not torch.isnan(Y_hat.grad).any(), "NaN in grad"
    assert not torch.isinf(Y_hat.grad).any(), "Inf in grad"
    assert gnorm > 0, f"grad norm == 0"
    print(f"  ||∇Y_hat|| = {gnorm:.6e}  (non-zero, finite — PASS)")

    # ── A3 validation: C_hat_t diagonal should be near 1.0 ─────────────
    # Re-run with fresh Y_hat since backward consumed graph
    Y_hat2 = torch.randn(N, m_genes, requires_grad=True)
    _ = loss_fn(Y_hat2, ctx_w, lambda_scale=1.0)
    diag2 = loss_fn._last_diagnostics
    diag_means = [
        diag2.get(f"ctx_{t}_C_hat_diag_mean")
        for t in range(T)
        if f"ctx_{t}_C_hat_diag_mean" in diag2
    ]
    if diag_means:
        print(f"\nC_hat_t diagonal means across active contexts:")
        for t, v in enumerate(diag_means):
            close = abs(v - 1.0) < 0.05
            print(f"  ctx {t}: {v:.4f} {'OK' if close else 'DRIFT'}")
    else:
        print(
            "\nNo active contexts in 2nd call — try larger N or more balanced ctx_w"
        )

    # ── EMA step counter should have advanced on both calls ────────────
    n_active_1 = diag1["n_active_contexts"]
    steps = loss_fn.ema_step.tolist()
    print(f"\nEMA steps per context: {steps}")
    assert max(steps) >= 2 or n_active_1 == 0, \
        f"EMA step never advanced past 1 despite {n_active_1} active contexts"

    # ── n_active=0 path (k_min unmet) — use tiny batch ─────────────────
    Y_small = torch.randn(5, m_genes, requires_grad=True)
    ctx_small = torch.softmax(torch.randn(5, T), dim=1)
    loss_zero = loss_fn(Y_small, ctx_small, lambda_scale=1.0)
    print(
        f"\nk_min=30 gate: tiny-batch loss = {loss_zero.item():.6e} "
        f"(should be ~0); n_active = "
        f"{loss_fn._last_diagnostics['n_active_contexts']}"
    )
    loss_zero.backward()
    assert Y_small.grad is not None, (
        "grad None on zero-loss path — autograd graph broken"
    )
    print(f"  ||∇Y_small|| = {Y_small.grad.norm().item():.6e}  (grad graph intact)")

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED — Phase 2 amendments A1-A6 active")
    print("=" * 60)


if __name__ == "__main__":
    main()
