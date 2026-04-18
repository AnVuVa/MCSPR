import numpy as np
from collections import defaultdict
from typing import Dict, Optional


class DriftTracker:

    def __init__(
        self,
        M: np.ndarray,
        intervention_threshold: float = 0.05,
    ):
        # Compute per-gene loading magnitude (row norms of M)
        self.loading_magnitude = np.linalg.norm(M, axis=1)  # (m,)

        # Compute quartile boundaries at 25/50/75 percentiles
        bounds = np.percentile(self.loading_magnitude, [25, 50, 75])
        # Assign quartile labels: 0=Q1 lowest load, 3=Q4 highest load
        self.quartile_labels = np.digitize(self.loading_magnitude, bounds)

        self.intervention_threshold = intervention_threshold
        self.history: Dict[str, list] = defaultdict(list)
        self.baseline_q1_mse: Optional[float] = None

    def update(
        self,
        epoch: int,
        Y_pred: np.ndarray,
        Y_true: np.ndarray,
        mcspr_loss: float,
    ) -> None:
        per_gene_mse = np.mean((Y_pred - Y_true) ** 2, axis=0)  # (m,)

        quartile_mse: Dict[str, float] = {}
        for q in range(4):
            mask = self.quartile_labels == q
            if mask.sum() > 0:
                quartile_mse[f"Q{q + 1}"] = float(np.mean(per_gene_mse[mask]))

        self.history["epoch"].append(epoch)
        self.history["mcspr_loss"].append(mcspr_loss)
        self.history["mse_per_quartile"].append(quartile_mse)

        # Set baseline at first epoch where mcspr_loss > 0
        if self.baseline_q1_mse is None and mcspr_loss > 0:
            self.baseline_q1_mse = quartile_mse.get("Q1", 0.0)

    def check_drift(self) -> Dict:
        if self.baseline_q1_mse is None or self.baseline_q1_mse == 0:
            return {
                "drift_detected": False,
                "q1_mse_delta_fraction": 0.0,
                "baseline_q1_mse": self.baseline_q1_mse,
                "current_q1_mse": None,
                "intervention_recommended": None,
                "note": (
                    "Baseline not yet established "
                    "(MCSPR loss has not been > 0)"
                ),
            }

        if not self.history["mse_per_quartile"]:
            return {
                "drift_detected": False,
                "q1_mse_delta_fraction": 0.0,
                "baseline_q1_mse": self.baseline_q1_mse,
                "current_q1_mse": None,
                "intervention_recommended": None,
                "note": "No history recorded yet",
            }

        current_q1_mse = self.history["mse_per_quartile"][-1].get("Q1", 0.0)
        delta = (current_q1_mse - self.baseline_q1_mse) / self.baseline_q1_mse
        drift_detected = delta > self.intervention_threshold

        result: Dict = {
            "drift_detected": drift_detected,
            "q1_mse_delta_fraction": float(delta),
            "baseline_q1_mse": float(self.baseline_q1_mse),
            "current_q1_mse": float(current_q1_mse),
            "intervention_recommended": None,
            "note": None,
        }

        if drift_detected:
            result["intervention_recommended"] = "head_only_mcspr"
            result["note"] = (
                "Stop gradient at h_i. Apply MCSPR only to prediction head. "
                "Zero new hyperparameters."
            )

        return result

    def report(self) -> Dict:
        drift_check = self.check_drift()

        # Pearson correlation between Q1 MSE and MCSPR loss
        # Negative = drift confirmed; near zero = safe
        q1_mse_series = [
            entry.get("Q1", 0.0)
            for entry in self.history["mse_per_quartile"]
        ]
        mcspr_series = self.history["mcspr_loss"]

        anti_corr: Optional[float] = None
        if len(q1_mse_series) >= 2 and len(mcspr_series) >= 2:
            from scipy.stats import pearsonr

            corr, _ = pearsonr(q1_mse_series, mcspr_series)
            anti_corr = float(corr) if not np.isnan(corr) else None

        return {
            "epochs": list(self.history["epoch"]),
            "mcspr_loss": list(self.history["mcspr_loss"]),
            "mse_per_quartile": list(self.history["mse_per_quartile"]),
            "final_drift_check": drift_check,
            "anti_correlation_q1_mcspr": anti_corr,
        }
