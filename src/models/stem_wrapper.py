"""
STEM preprocessing wrapper for unified benchmark protocol.

Problem: STEM uses log2(count+1), we use log1p(count/sum).
Directly feeding our compressed distribution into STEM's schedule
causes SNR collapse in first 10% of timesteps (proven analytically).

Solution: Per-gene z-score standardization before diffusion.
  - Forces Var[y_tilde] = 1.0 per gene regardless of upstream normalization
  - STEM's schedule was calibrated for Var ~ 1.0
  - Unstandardize after reverse diffusion
  - Clamp AFTER unstandardization (not before)

Critical clamp ordering:
  Zero boundary in log1p(count/sum) space = 0
  Zero boundary in standardized space = -mu_j / sigma_j (gene-specific, negative)
  Clamping at 0 in standardized space destroys valid near-zero predictions.
  Must clamp at 0 AFTER unstandardization.

STEM audit confirmed:
  - stem_sample.py line 66: clip_denoised=False
  - NO boundary enforcement in sampling pipeline
  - Must add torch.clamp(samples, min=0.0) after expm1 equivalent
  - For our protocol (log1p not log2): clamp after unstandardize

Protocol (8 steps, locked):
  1. log1p(count/sum) normalization (same as all other models)
  2. Compute mu_j, sigma_j on training fold only
  3. Standardize: y_tilde = (y - mu) / sigma
  4. Verify schedule: forward T steps, check Var[y_T] ~ 1.0
  5. Run STEM's reverse diffusion on standardized conditioning
  6. Unstandardize: y_hat = y_tilde_hat * sigma + mu
  7. Clamp: y_hat = max(y_hat, 0.0) [after unstandardization]
  8. Evaluate all metrics in log1p(count/sum) space
"""

import torch
import numpy as np
from typing import Optional


class STEMPreprocessingWrapper:
    """
    Wraps STEM's training and sampling pipeline with per-gene standardization.

    Usage:
        wrapper = STEMPreprocessingWrapper()
        wrapper.fit(Y_train)           # compute mu_j, sigma_j on training data
        Y_tilde = wrapper.standardize(Y)
        # ... run STEM training on Y_tilde ...
        Y_hat = wrapper.unstandardize_and_clamp(Y_tilde_hat)

    Args:
        epsilon: Small constant for numerical stability in sigma denominator
        verify_schedule: If True, run schedule verification on fit()
        T_steps: Number of diffusion timesteps (default 1000, matches STEM)
    """

    def __init__(
        self,
        epsilon: float = 1e-6,
        verify_schedule: bool = True,
        T_steps: int = 1000,
    ):
        self.epsilon = epsilon
        self.verify_schedule = verify_schedule
        self.T_steps = T_steps
        self.mu: Optional[np.ndarray] = None     # (m,)
        self.sigma: Optional[np.ndarray] = None   # (m,)
        self._fitted = False

    def fit(self, Y_train: np.ndarray) -> "STEMPreprocessingWrapper":
        """
        Compute per-gene mean and std from training data.

        Args:
            Y_train: (N_train, m) log1p(count/sum) normalized expression
                     TRAINING DATA ONLY -- never fit on test data

        Returns:
            self (for chaining)
        """
        assert Y_train.ndim == 2, f"Expected (N, m), got {Y_train.shape}"

        self.mu = Y_train.mean(axis=0)                    # (m,)
        self.sigma = Y_train.std(axis=0) + self.epsilon    # (m,)

        # Verify no zero-sigma genes (would cause division by zero)
        n_zero_sigma = (self.sigma <= self.epsilon * 2).sum()
        if n_zero_sigma > 0:
            print(f"  WARNING: {n_zero_sigma} genes have near-zero variance. "
                  f"These genes will have sigma clamped to epsilon={self.epsilon}.")
            self.sigma = np.maximum(self.sigma, self.epsilon)

        self._fitted = True

        print(f"STEM wrapper fitted on {Y_train.shape[0]} spots, "
              f"{Y_train.shape[1]} genes")
        print(f"  Gene sigma range: [{self.sigma.min():.4f}, {self.sigma.max():.4f}]")
        print(f"  After standardization: expected Var ~ 1.0 per gene")

        if self.verify_schedule:
            self._verify_schedule(Y_train)

        return self

    def _verify_schedule(self, Y_train: np.ndarray, n_sample: int = 500):
        """
        Verify that STEM's linear/cosine schedule reaches Var ~ 1.0 by step T
        when applied to the standardized distribution.

        Standardization ensures Var[y_tilde] = 1.0. A standard DDPM linear
        schedule designed for Var[data] = 1.0 should reach Var[y_T] ~ 1.0
        (i.e., pure noise) by step T.

        If Var[y_T] >> 1.0: schedule reaches noise too fast (shorten T or reduce beta_max)
        If Var[y_T] << 1.0: schedule too slow (increase beta_max or T)
        """
        # Simulate linear beta schedule (STEM default)
        beta_start, beta_end = 1e-4, 0.02
        betas = np.linspace(beta_start, beta_end, self.T_steps)
        alphas = 1.0 - betas
        alpha_bars = np.cumprod(alphas)

        # Sample n_sample spots, standardize, run forward to T
        idx = np.random.choice(len(Y_train), min(n_sample, len(Y_train)),
                               replace=False)
        Y_sample = self.standardize(Y_train[idx])    # (n, m)

        # Forward process at T: q(y_T | y_0) = N(sqrt(alpha_bar_T) * y_0, (1-alpha_bar_T)I)
        alpha_bar_T = alpha_bars[-1]
        noise = np.random.randn(*Y_sample.shape)
        Y_T = np.sqrt(alpha_bar_T) * Y_sample + np.sqrt(1 - alpha_bar_T) * noise

        var_T = Y_T.var(axis=0).mean()
        print(f"\nSchedule verification (linear, T={self.T_steps}):")
        print(f"  Var[y_tilde] = {Y_sample.var(axis=0).mean():.4f}  (should be ~ 1.0)")
        print(f"  Var[y_T]     = {var_T:.4f}  (should be ~ 1.0 for pure noise)")
        print(f"  alpha_bar_T  = {alpha_bar_T:.6f}  (should be ~ 0.0)")

        if abs(var_T - 1.0) < 0.1:
            print(f"  Schedule status: OK -- schedule reaches noise correctly")
        else:
            print(f"  Schedule status: WARNING -- Var[y_T]={var_T:.4f} deviates from 1.0")
            print(f"  Action: adjust beta_max or T before training STEM")

    def standardize(self, Y: np.ndarray) -> np.ndarray:
        """
        Standardize: y_tilde = (y - mu) / sigma

        Args:
            Y: (N, m) log1p-normalized expression
        Returns:
            Y_tilde: (N, m) standardized, Var ~ 1.0 per gene
        """
        self._check_fitted()
        return (Y - self.mu) / self.sigma

    def unstandardize_and_clamp(
        self,
        Y_tilde_hat: np.ndarray,
        clamp_min: float = 0.0,
    ) -> np.ndarray:
        """
        Unstandardize then clamp to valid expression range.

        CRITICAL ORDER: clamp AFTER unstandardization.
        Zero in log1p-space = 0.0
        Zero in standardized space = -mu_j / sigma_j (gene-specific, negative)
        Clamping at 0 in standardized space would destroy predictions where
        standardized value is negative but maps to a valid positive raw count.

        Args:
            Y_tilde_hat: (N, m) standardized predictions from STEM reverse diffusion
            clamp_min:   Minimum value in log1p-space (default 0.0)

        Returns:
            Y_hat: (N, m) in log1p(count/sum) space, non-negative
        """
        self._check_fitted()
        Y_hat = Y_tilde_hat * self.sigma + self.mu    # unstandardize
        Y_hat = np.maximum(Y_hat, clamp_min)           # clamp to [0, inf)
        return Y_hat

    def unstandardize_and_clamp_torch(
        self,
        Y_tilde_hat: torch.Tensor,
        clamp_min: float = 0.0,
    ) -> torch.Tensor:
        """Torch version for GPU inference."""
        self._check_fitted()
        mu = torch.tensor(self.mu, dtype=Y_tilde_hat.dtype,
                          device=Y_tilde_hat.device)
        sigma = torch.tensor(self.sigma, dtype=Y_tilde_hat.dtype,
                             device=Y_tilde_hat.device)
        Y_hat = Y_tilde_hat * sigma + mu
        return torch.clamp(Y_hat, min=clamp_min)

    def save(self, path: str):
        """Save fitted parameters for reproducibility."""
        self._check_fitted()
        np.savez(path, mu=self.mu, sigma=self.sigma,
                 epsilon=np.array([self.epsilon]),
                 T_steps=np.array([self.T_steps]))
        print(f"STEM wrapper saved to {path}")

    @classmethod
    def load(cls, path: str) -> "STEMPreprocessingWrapper":
        """Load fitted parameters."""
        data = np.load(path)
        wrapper = cls(
            epsilon=float(data["epsilon"][0]),
            T_steps=int(data["T_steps"][0]),
            verify_schedule=False,
        )
        wrapper.mu = data["mu"]
        wrapper.sigma = data["sigma"]
        wrapper._fitted = True
        return wrapper

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError(
                "STEMPreprocessingWrapper is not fitted. "
                "Call wrapper.fit(Y_train) before standardize/unstandardize."
            )
