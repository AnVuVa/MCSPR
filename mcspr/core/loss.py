"""MCSPR loss — MCSPR_FINAL_LOCKED_V2.md File 3 (amendments A1–A6).

Interface: forward(Y_hat, context_weights, lambda_scale) → L_MCSPR (scalar).
Diagnostics available as side-channel attribute ``self._last_diagnostics``.
"""

import torch
import torch.nn as nn
from typing import Dict


class MCSPRLoss(nn.Module):

    def __init__(
        self,
        M_pinv: torch.Tensor,
        C_prior: torch.Tensor,
        n_contexts: int,
        k_min: float = 30.0,
        tau: float = 1.0e-4,
        beta: float = 0.9,
        lambda_max: float = 0.1,
    ):
        super().__init__()
        B = int(M_pinv.shape[0])
        self.n_modules = B
        self.n_contexts = n_contexts
        self.k_min = k_min
        self.tau = tau
        self.beta = beta
        self.lambda_max = lambda_max

        self.register_buffer("M_pinv", M_pinv)
        self.register_buffer("C_prior", C_prior)
        self.register_buffer("ema_mu", torch.zeros(n_contexts, B))
        self.register_buffer("ema_Sigma", torch.zeros(n_contexts, B, B))
        self.register_buffer("ema_sigma_sq", torch.zeros(n_contexts, B))
        self.register_buffer(
            "ema_initialized", torch.zeros(n_contexts, dtype=torch.bool)
        )
        self.register_buffer(
            "ema_step", torch.zeros(n_contexts, dtype=torch.long)
        )

        self._last_diagnostics: Dict = {}

    def forward(
        self,
        Y_hat: torch.Tensor,
        context_weights: torch.Tensor,
        lambda_scale: float = 1.0,
    ) -> torch.Tensor:
        diagnostics: Dict = {}

        # Project to NMF latent space — gradient flows through Y_hat
        Z_hat = Y_hat @ self.M_pinv.T  # (N, B)
        B_dim = Z_hat.shape[1]

        total_loss = torch.tensor(
            0.0, device=Y_hat.device, dtype=Y_hat.dtype
        )
        n_active = 0

        for t in range(self.n_contexts):
            w_t = context_weights[:, t]  # (N,) frozen
            eff_n = w_t.sum().item()

            if eff_n < self.k_min:
                diagnostics[f"ctx_{t}_skipped"] = True
                continue

            n_active += 1

            # Per-context soft-weighted batch mean and covariance (HAS grad)
            w_sum = w_t.sum()
            w_norm = (w_t / w_sum).unsqueeze(1)  # (N, 1)
            mu_batch = (w_t.unsqueeze(1) * Z_hat).sum(0) / w_sum  # (B,)
            z_c = Z_hat - mu_batch.unsqueeze(0)  # (N, B) centered
            Sigma_batch_t = (z_c * w_norm).T @ z_c  # (B, B) HAS grad

            # ── Buffer update (no_grad); historical state severed ────────
            with torch.no_grad():
                if not self.ema_initialized[t]:
                    # ★ A1 — zero-consistent warm start
                    self.ema_Sigma[t] = (
                        (1.0 - self.beta) * Sigma_batch_t.detach()
                    )
                    self.ema_mu[t] = (1.0 - self.beta) * mu_batch.detach()
                    self.ema_initialized[t] = True
                else:
                    self.ema_Sigma[t] = (
                        self.beta * self.ema_Sigma[t].detach()
                        + (1.0 - self.beta) * Sigma_batch_t.detach()
                    )
                    self.ema_mu[t] = (
                        self.beta * self.ema_mu[t].detach()
                        + (1.0 - self.beta) * mu_batch.detach()
                    )
                self.ema_step[t] += 1
                self.ema_sigma_sq[t] = torch.diag(self.ema_Sigma[t])

            # ── ★ A2 — gradient-carrying EMA recompute (OUTSIDE no_grad) ─
            step = self.ema_step[t].item()
            if step > 1:
                # Historical buffer detached; current batch flows via (1-β)
                ema_Sigma_with_grad = (
                    self.beta * self.ema_Sigma[t].detach()
                    + (1.0 - self.beta) * Sigma_batch_t
                )
            else:
                # First-ever observation of context t — warm-start only
                ema_Sigma_with_grad = (1.0 - self.beta) * Sigma_batch_t

            # Bias correction (Adam-style)
            bias_correction = 1.0 - self.beta ** step
            Sigma_unbiased = ema_Sigma_with_grad / bias_correction

            # Denominator: detached EMA std (no gradient)
            std_a = torch.sqrt(
                Sigma_unbiased.diag().detach() + self.tau
            )  # (B,) no grad
            denom = std_a.unsqueeze(1) * std_a.unsqueeze(0)  # (B, B) no grad

            # ── ★ A3 — C_hat_t numerator = Sigma_unbiased (not Sigma_batch_t) ──
            # Diagonal = Σ_ii / (Σ_ii + τ) ≈ 1.0, valid correlation matrix
            C_hat_t = Sigma_unbiased / denom
            C_hat_t = torch.clamp(C_hat_t, -1.0, 1.0)

            # ── ★ A4 — B² normalization (dim-invariant mean sq residual) ──
            loss_t = (1.0 / (eff_n * B_dim * B_dim)) * torch.sum(
                (C_hat_t.float() - self.C_prior[t].float()) ** 2
            )

            diagnostics[f"ctx_{t}_eff_n"] = eff_n
            diagnostics[f"ctx_{t}_loss"] = loss_t.item()
            diagnostics[f"ctx_{t}_C_hat_diag_mean"] = (
                torch.diag(C_hat_t).mean().item()
            )

            total_loss = total_loss + loss_t

        diagnostics["n_active_contexts"] = n_active
        diagnostics["effective_lambda"] = self.lambda_max * lambda_scale
        self._last_diagnostics = diagnostics

        if n_active == 0:
            # Zero loss that preserves autograd graph through Y_hat
            return Y_hat.sum() * 0.0

        # ── ★ A5 + A6 — fixed T denominator, 1/(1-β) gradient restoration ──
        L_MCSPR = (
            (self.lambda_max * lambda_scale)
            * total_loss
            / self.n_contexts
        )
        L_MCSPR = L_MCSPR / (1.0 - self.beta)

        return L_MCSPR
