import torch
import torch.nn as nn
from typing import Dict, Tuple


class MCSPRLoss(nn.Module):

    def __init__(
        self,
        M_pinv: torch.Tensor,
        C_prior: torch.Tensor,
        n_modules: int,
        n_contexts: int,
        k_min: float = 30.0,
        tau: float = 1e-6,
        beta: float = 0.9,
        lambda_max: float = 0.1,
    ):
        super().__init__()
        self.n_modules = n_modules
        self.n_contexts = n_contexts
        self.k_min = k_min
        self.tau = tau
        self.beta = beta
        self.lambda_max = lambda_max

        self.register_buffer("M_pinv", M_pinv)
        self.register_buffer("C_prior", C_prior)
        self.register_buffer("ema_mu", torch.zeros(n_contexts, n_modules))
        self.register_buffer("ema_sigma_sq", torch.ones(n_contexts, n_modules))
        self.register_buffer(
            "ema_Sigma",
            torch.eye(n_modules).unsqueeze(0).expand(n_contexts, -1, -1).clone(),
        )
        self.register_buffer(
            "ema_initialized", torch.zeros(n_contexts, dtype=torch.bool)
        )

    def forward(
        self,
        Y_hat: torch.Tensor,
        context_weights: torch.Tensor,
        lambda_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict]:
        diagnostics: Dict = {}

        # Step 1 — Project to module space
        Z_hat = Y_hat @ self.M_pinv.T  # (n, B), gradient flows through this

        total_loss = torch.tensor(0.0, device=Y_hat.device, dtype=Y_hat.dtype)
        n_active_contexts = 0

        # Step 2 — For each context t
        for t in range(self.n_contexts):
            # a) Get soft weights
            w_t = context_weights[:, t]  # (n,)

            # b) Compute effective n
            eff_n = w_t.sum().item()

            # c) If eff_n < k_min: skip context. NEVER sample with replacement.
            if eff_n < self.k_min:
                diagnostics[f"ctx_{t}_skipped"] = True
                continue

            n_active_contexts += 1

            # d) Soft-weighted mean
            mu_t = (w_t.unsqueeze(1) * Z_hat).sum(0) / w_t.sum()  # (B,)

            # e) Soft-weighted covariance
            z_centered = Z_hat - mu_t.unsqueeze(0)  # (n, B)
            w_norm = (w_t / w_t.sum()).unsqueeze(1)  # (n, 1)
            Sigma_t = (z_centered * w_norm).T @ z_centered  # (B, B)

            # f) EMA update — on MOMENTS, NOT on normalized correlations.
            #    CRITICAL: Never do ema_C[t] = beta * ema_C[t] + (1-beta) * C_hat_t.
            #    E[ratio] != ratio of E[]. Always EMA on Sigma, then normalize after.
            with torch.no_grad():
                if not self.ema_initialized[t]:
                    self.ema_mu[t] = mu_t.detach()
                    self.ema_Sigma[t] = Sigma_t.detach()
                    self.ema_sigma_sq[t] = torch.diag(Sigma_t).detach()
                    self.ema_initialized[t] = True
                else:
                    self.ema_mu[t] = (
                        self.beta * self.ema_mu[t]
                        + (1 - self.beta) * mu_t.detach()
                    )
                    self.ema_Sigma[t] = (
                        self.beta * self.ema_Sigma[t]
                        + (1 - self.beta) * Sigma_t.detach()
                    )
                    self.ema_sigma_sq[t] = torch.diag(self.ema_Sigma[t])

            # g) Normalized correlation using EMA variance for denominator (stable)
            #    and current-batch Sigma_t for numerator (carries gradient).
            std_a = torch.sqrt(
                self.ema_sigma_sq[t] + self.tau
            )  # (B,) — no gradient (from buffer)
            denom = (
                std_a.unsqueeze(1) * std_a.unsqueeze(0)
            )  # (B, B) — no gradient
            C_hat_t = Sigma_t / (denom + self.tau)  # (B, B) — HAS gradient via Sigma_t
            C_hat_t = torch.clamp(C_hat_t, -1.0, 1.0)

            # h) Per-context Frobenius loss
            loss_t = (1.0 / eff_n) * torch.sum(
                (C_hat_t - self.C_prior[t]) ** 2
            )

            # i) Log diagnostics
            diagnostics[f"ctx_{t}_eff_n"] = eff_n
            diagnostics[f"ctx_{t}_loss"] = loss_t.item()
            diagnostics[f"ctx_{t}_C_hat_diag_mean"] = (
                torch.diag(C_hat_t).mean().item()
            )

            total_loss = total_loss + loss_t

        # Step 3 — Aggregate
        if n_active_contexts > 0:
            total_loss = (
                (self.lambda_max * lambda_scale) * total_loss / n_active_contexts
            )
        diagnostics["n_active_contexts"] = n_active_contexts
        diagnostics["effective_lambda"] = self.lambda_max * lambda_scale
        return total_loss, diagnostics
