"""Training loop for TRIPLEX and TRIPLEX+MCSPR.

MCSPR loss attaches to preds['fusion'] — the fusion token prediction.
The fusion token integrates all three views; applying MCSPR here
maximizes gradient coverage.
"""

import json
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Optional

from src.losses.triplex_loss import TRIPLEXLoss
from src.training.evaluate import evaluate_fold


def train_one_fold(
    model: nn.Module,
    train_loader,
    val_loader,
    config: dict,
    fold_idx: int,
    output_dir: str,
    device: str = "cuda",
    mcspr_artifacts: Optional[Dict] = None,
    gene_names=None,
    dry_run: bool = False,
) -> Dict:
    """Train one fold. Shared by both TRIPLEX and TRIPLEX+MCSPR.

    Args:
        model: TRIPLEX model instance
        mcspr_artifacts: if not None, dict with:
            M_pinv (np.ndarray), C_prior (np.ndarray),
            n_modules (int), n_contexts (int)
            Enables MCSPR regularization.
        dry_run: if True, run one batch and exit.
    """
    tc = config.get("training", {})
    mc = config.get("mcspr", {})

    lr = tc.get("lr", 0.0001)
    max_epochs = tc.get("max_epochs", 200)
    patience = tc.get("early_stopping_patience", 20)
    step_size = tc.get("step_size", 50)
    gamma = tc.get("gamma", 0.9)
    alpha = tc.get("alpha", 0.5)
    use_amp = tc.get("mixed_precision", True)

    model = model.to(device)

    # Loss functions
    triplex_loss_fn = TRIPLEXLoss(alpha=alpha)

    # MCSPR setup
    mcspr_loss_fn = None
    lambda_scheduler = None

    if mcspr_artifacts is not None:
        from mcspr import MCSPRLoss, LambdaScheduler

        mcspr_loss_fn = MCSPRLoss(
            M_pinv=torch.tensor(
                mcspr_artifacts["M_pinv"], dtype=torch.float32
            ).to(device),
            C_prior=torch.tensor(
                mcspr_artifacts["C_prior"], dtype=torch.float32
            ).to(device),
            n_modules=mcspr_artifacts["n_modules"],
            n_contexts=mcspr_artifacts["n_contexts"],
            k_min=mc.get("k_min", 30.0),
            tau=mc.get("tau", 1e-6),
            beta=mc.get("beta", 0.9),
            lambda_max=mc.get("lambda_max", 0.1),
        ).to(device)

        lambda_scheduler = LambdaScheduler(
            warmup_epochs=mc.get("warmup_epochs", 5),
            ramp_epochs=mc.get("ramp_epochs", 10),
        )

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma
    )

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda") if use_amp and device == "cuda" else None

    # Output directory
    fold_dir = Path(output_dir) / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # Training state
    best_pcc_m = -float("inf")
    epochs_without_improvement = 0
    training_log = []

    for epoch in range(max_epochs):
        model.train()
        epoch_losses = []
        epoch_mcspr_losses = []
        epoch_n_active = []

        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            b = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            optimizer.zero_grad()

            if use_amp and scaler is not None:
                with torch.amp.autocast("cuda"):
                    preds, tokens = model(b)
                    y = b["expression"]
                    triplex_loss, components = triplex_loss_fn(preds, y)

                    if mcspr_loss_fn is not None:
                        lambda_scale = lambda_scheduler.get_scale(epoch)
                        context_w = b["context_weights"]
                        mcspr_loss, mcspr_diag = mcspr_loss_fn(
                            Y_hat=preds["fusion"],
                            context_weights=context_w,
                            lambda_scale=lambda_scale,
                        )
                        total_loss = triplex_loss + mcspr_loss
                        epoch_mcspr_losses.append(mcspr_loss.item())
                        epoch_n_active.append(
                            mcspr_diag.get("n_active_contexts", 0)
                        )
                    else:
                        total_loss = triplex_loss

                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                preds, tokens = model(b)
                y = b["expression"]
                triplex_loss, components = triplex_loss_fn(preds, y)

                if mcspr_loss_fn is not None:
                    lambda_scale = lambda_scheduler.get_scale(epoch)
                    context_w = b["context_weights"]
                    mcspr_loss, mcspr_diag = mcspr_loss_fn(
                        Y_hat=preds["fusion"],
                        context_weights=context_w,
                        lambda_scale=lambda_scale,
                    )
                    total_loss = triplex_loss + mcspr_loss
                    epoch_mcspr_losses.append(mcspr_loss.item())
                    epoch_n_active.append(
                        mcspr_diag.get("n_active_contexts", 0)
                    )
                else:
                    total_loss = triplex_loss

                total_loss.backward()
                optimizer.step()

            epoch_losses.append(total_loss.item())

            # Dry run: one batch, print diagnostics, exit
            if dry_run:
                print(f"\n=== DRY RUN — Fold {fold_idx}, Batch 0 ===")
                print(f"  TRIPLEX loss: {triplex_loss.item():.6f}")
                for k, v in components.items():
                    print(f"    {k}: {v:.6f}")

                # Check grad norm on target encoder
                te_params = [
                    p
                    for p in model.target_encoder.parameters()
                    if p.grad is not None
                ]
                if te_params:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        te_params, float("inf")
                    )
                    print(f"  Target encoder grad norm: {grad_norm:.6f}")

                if mcspr_loss_fn is not None:
                    print(f"\n  MCSPR loss: {mcspr_loss.item():.6f}")
                    print(f"  MCSPR diagnostics: {mcspr_diag}")
                    print(
                        f"  lambda_scale at epoch 0: "
                        f"{lambda_scheduler.get_scale(0)}"
                    )

                    # Verify grad flows through fusion head
                    fusion_params = [
                        p
                        for p in model.fusion_layer.parameters()
                        if p.grad is not None
                    ]
                    if fusion_params:
                        fg_norm = torch.nn.utils.clip_grad_norm_(
                            fusion_params, float("inf")
                        )
                        print(
                            f"  MCSPR grad flows through fusion head: "
                            f"norm={fg_norm:.6f}"
                        )

                return {"dry_run": True}

        lr_scheduler.step()

        # Epoch summary
        mean_loss = float(np.mean(epoch_losses))
        log_entry = {
            "epoch": epoch,
            "train_loss": mean_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }

        if epoch_mcspr_losses:
            log_entry["mcspr_loss"] = float(np.mean(epoch_mcspr_losses))
            log_entry["n_active_contexts"] = float(np.mean(epoch_n_active))

        # Validation
        val_metrics = evaluate_fold(
            model,
            val_loader,
            config,
            gene_names or [],
            device=device,
            mcspr_artifacts=mcspr_artifacts,
        )
        log_entry["val_pcc_m"] = val_metrics["pcc_m"]
        log_entry["val_pcc_h"] = val_metrics["pcc_h"]
        log_entry["val_mse"] = val_metrics["mse"]
        log_entry["val_mae"] = val_metrics["mae"]
        if val_metrics.get("smcs_overall") is not None:
            log_entry["val_smcs"] = val_metrics["smcs_overall"]

        training_log.append(log_entry)

        # Print progress
        mcspr_str = ""
        if "mcspr_loss" in log_entry:
            mcspr_str = (
                f" | MCSPR={log_entry['mcspr_loss']:.4f} "
                f"({log_entry['n_active_contexts']:.0f}ctx)"
            )
        print(
            f"  Epoch {epoch:3d} | loss={mean_loss:.4f} "
            f"| val_PCC(M)={val_metrics['pcc_m']:.4f} "
            f"| val_MSE={val_metrics['mse']:.4f}"
            f"{mcspr_str}"
        )

        # Early stopping
        if val_metrics["pcc_m"] > best_pcc_m:
            best_pcc_m = val_metrics["pcc_m"]
            epochs_without_improvement = 0
            # Save best model
            torch.save(
                model.state_dict(), str(fold_dir / "best_model.pt")
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"  Early stopping at epoch {epoch} "
                    f"(patience={patience})"
                )
                break

    # Save training log
    with open(str(fold_dir / "training_log.json"), "w") as f:
        json.dump(training_log, f, indent=2)

    # Load best model and final evaluation
    best_state = torch.load(str(fold_dir / "best_model.pt"), map_location=device)
    model.load_state_dict(best_state)
    final_metrics = evaluate_fold(
        model,
        val_loader,
        config,
        gene_names or [],
        device=device,
        mcspr_artifacts=mcspr_artifacts,
    )

    # Save final metrics
    with open(str(fold_dir / "metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2, default=str)

    return final_metrics
