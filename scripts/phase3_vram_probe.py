"""Phase 3 — TRIPLEX VRAM probe at batch_size=256 on HER2ST fold 0.

Loads TRIPLEX, constructs a real batch from fold 0 training loader,
runs one autocast forward + backward with MCSPR + NormalizedMSELoss,
records torch.cuda.max_memory_allocated(). Exits.

Waits for data/her2st/nmf/fold_0/C_prior.npy to exist (Phase 1 may
still be writing artifacts in parallel).
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def wait_for(path: Path, timeout_s: int = 300):
    t0 = time.time()
    while not path.exists():
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"Timed out waiting for {path}")
        time.sleep(1)


def main():
    assert torch.cuda.is_available(), "CUDA required for VRAM probe"

    cfg_path = "configs/her2st.yaml"
    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    # Spec v2 overrides
    config["n_folds"] = 4
    config.setdefault("training", {})["batch_size"] = 256
    config["training"]["use_normalized_mse"] = True
    config["training"]["num_workers"] = 0
    config["mcspr"]["enabled"] = True
    config["mcspr"]["n_modules"] = 10
    config["mcspr"]["lambda_max"] = 0.1  # probe only; Phase 4 determines real

    data_dir = config["data_dir"]
    dataset = config["dataset"]

    # Wait for fold 0 MCSPR artifacts (Phase 1 in parallel)
    fold0_dir = Path(data_dir) / "nmf" / "fold_0"
    print(f"[probe] waiting for {fold0_dir}/C_prior.npy ...", flush=True)
    wait_for(fold0_dir / "C_prior.npy", timeout_s=600)
    wait_for(fold0_dir / "M_pinv.npy", timeout_s=60)
    wait_for(fold0_dir / "gene_var.npy", timeout_s=60)
    time.sleep(2)  # ensure writes are flushed
    print("[probe] artifacts present.")

    from src.data.loaders import build_loaders
    from src.models.triplex import TRIPLEX
    from mcspr.core.loss import MCSPRLoss
    from src.losses.normalized_mse import NormalizedMSELoss

    samples = sorted(
        [p.stem for p in Path(data_dir, "barcodes").glob("*.csv")]
    )

    # Build loader for fold 0 (patch-based patient split A,B test → 24 train)
    train_loader, _val = build_loaders(
        data_dir, dataset, fold_idx=0,
        config=config, sample_names=samples,
    )

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Build TRIPLEX
    model = TRIPLEX(config, n_genes=config.get("n_genes", 300)).to(device)

    # Build MCSPR + NormalizedMSELoss from fold 0 artifacts
    M_pinv = np.load(fold0_dir / "M_pinv.npy")
    C_prior = np.load(fold0_dir / "C_prior.npy")
    gene_var = np.load(fold0_dir / "gene_var.npy")

    print(
        f"[probe] M_pinv={M_pinv.shape} C_prior={C_prior.shape} "
        f"gene_var={gene_var.shape}"
    )

    mcspr_loss_fn = MCSPRLoss(
        M_pinv=torch.tensor(M_pinv, dtype=torch.float32).to(device),
        C_prior=torch.tensor(C_prior, dtype=torch.float32).to(device),
        n_contexts=config["mcspr"]["n_contexts"],
        k_min=config["mcspr"]["k_min"],
        tau=config["mcspr"].get("tau", 1e-4),
        beta=config["mcspr"].get("beta", 0.9),
        lambda_max=config["mcspr"]["lambda_max"],
    ).to(device)

    mse_loss_fn = NormalizedMSELoss(gene_var).to(device)

    # Fetch ONE batch and run forward+backward with AMP (matches training)
    batch = next(iter(train_loader))
    batch_gpu = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }

    # Sanity: print batch shapes
    for k, v in batch_gpu.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")

    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    opt.zero_grad(set_to_none=True)

    model.train()
    with autocast():
        preds, _tokens = model(batch_gpu)
        Y_hat = preds.get("output", preds["fusion"])
        Y_true = batch_gpu["expression"]
        mse = mse_loss_fn(Y_hat, Y_true)
        mcspr = mcspr_loss_fn(
            Y_hat, batch_gpu["context_weights"], lambda_scale=1.0
        )
        total = mse + mcspr

    scaler.scale(total).backward()

    peak_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)
    alloc_mib = torch.cuda.memory_allocated() / (1024 ** 2)
    total_mib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)

    print("\n" + "=" * 60)
    print(f"BATCH_SIZE = 256  fold 0  (TRIPLEX + MCSPR + NormalizedMSELoss)")
    print(f"Peak VRAM allocated:   {peak_mib:8.1f} MiB")
    print(f"Current VRAM alloc:    {alloc_mib:8.1f} MiB")
    print(f"GPU total capacity:    {total_mib:8.1f} MiB")
    print(f"Headroom at peak:      {total_mib - peak_mib:8.1f} MiB")
    print(f"MSE loss = {mse.item():.6f}")
    print(f"MCSPR loss = {mcspr.item():.6f}")
    print(f"MCSPR diag: n_active = "
          f"{mcspr_loss_fn._last_diagnostics.get('n_active_contexts')}")
    print("=" * 60)

    # Decision rule per user:
    # peak <= 14GB:   batch=256
    # peak 14-15GB:   batch=192
    # peak >15GB:     batch=128 (accept k_min risk)
    if peak_mib <= 14 * 1024:
        rec = 256
    elif peak_mib <= 15 * 1024:
        rec = 192
    else:
        rec = 128
    print(f"\nRECOMMENDED batch_size = {rec}")

    import json
    Path("results/v2").mkdir(parents=True, exist_ok=True)
    with open("results/v2/phase3_vram_probe.json", "w") as f:
        json.dump(
            {
                "probed_batch_size": 256,
                "peak_mib": peak_mib,
                "current_mib": alloc_mib,
                "total_mib": total_mib,
                "mse_loss": float(mse.item()),
                "mcspr_loss": float(mcspr.item()),
                "recommended_batch_size": rec,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
