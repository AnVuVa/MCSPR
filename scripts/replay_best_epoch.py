"""Replay MCSPR training_log.json files: compare current best checkpoint epoch
against best post-ramp epoch (gated to epoch >= warmup + ramp_epochs).

Detects folds whose saved best_model.pt reflects a warmup-era epoch — those are
candidates for re-run with a post-ramp gating rule on checkpoint selection.

No GPU, no retraining. Pure log analysis.
"""
import glob
import json
from pathlib import Path

WARMUP = 5
RAMP = 10
POST_RAMP_START = WARMUP + RAMP   # epoch 15 — first epoch at full lambda

# Real layout: results/{arch}/her2st/fold_N/training_log.json
# Lambda selection layout: results/lambda_selection/her2st/lambda_*/fold_99/...
log_files = sorted(
    glob.glob("results/stnet_mcspr/*/fold_*/training_log.json") +
    glob.glob("results/triplex_mcspr/*/fold_*/training_log.json") +
    glob.glob("results/histogene_mcspr/*/fold_*/training_log.json") +
    glob.glob("results/lambda_selection/*/lambda_*/fold_*/training_log.json")
)
print(f"Found {len(log_files)} MCSPR training logs\n")

results = []
for log_path in log_files:
    with open(log_path) as f:
        log = json.load(f)
    if not isinstance(log, list) or not log:
        print(f"  SKIP {log_path}: unexpected format")
        continue

    epochs = [e.get("epoch", i) for i, e in enumerate(log)]
    val_pccs = [e.get("val_pcc_m", float("-inf")) for e in log]
    lambda_vals = [e.get("lambda_scale", None) for e in log]

    # (a) best at ANY epoch
    best_any_idx = max(range(len(val_pccs)), key=lambda i: val_pccs[i])
    best_any_pcc = val_pccs[best_any_idx]
    best_any_ep = epochs[best_any_idx]

    # (b) best at epoch >= POST_RAMP_START
    post_ramp = [(i, val_pccs[i]) for i in range(len(val_pccs))
                 if epochs[i] >= POST_RAMP_START]
    if post_ramp:
        best_pr_idx, best_pr_pcc = max(post_ramp, key=lambda x: x[1])
        best_pr_ep = epochs[best_pr_idx]
    else:
        best_pr_pcc = float("-inf")
        best_pr_ep = None

    delta = best_pr_pcc - best_any_pcc if best_pr_ep is not None else None
    gating_changes_ckpt = (best_pr_ep is not None and best_pr_ep != best_any_ep)

    lam_at_best = lambda_vals[best_any_idx]
    label = log_path.replace("results/", "").replace("/training_log.json", "")

    if best_pr_ep is None:
        status = "NO POST-RAMP DATA (early-stop before epoch 15)"
    elif not gating_changes_ckpt:
        status = "OK (current ckpt already post-ramp)"
    elif abs(delta) < 0.005:
        status = "COSMETIC (gating changes ep but Δ<0.005)"
    elif delta > 0:
        status = "GATING HELPS (post-ramp better, re-run with gate)"
    else:
        status = "GATING HURTS (post-ramp worse — best is pre-ramp)"

    print(f"{label}")
    print(f"  total_epochs={len(val_pccs)}  post_ramp_epochs={len(post_ramp)}")
    print(f"  Best ANY  : epoch {best_any_ep:>3}  pcc={best_any_pcc:.4f}  "
          f"λ_at_save={lam_at_best}")
    if best_pr_ep is not None:
        print(f"  Best ≥{POST_RAMP_START}  : epoch {best_pr_ep:>3}  pcc={best_pr_pcc:.4f}  "
              f"Δ={delta:+.4f}")
    print(f"  → {status}")
    print()

    results.append({
        "fold": label,
        "total_epochs": len(val_pccs),
        "post_ramp_epochs": len(post_ramp),
        "best_any_epoch": best_any_ep,
        "best_any_pcc": best_any_pcc,
        "lambda_at_best": lam_at_best,
        "best_post_ramp_epoch": best_pr_ep,
        "best_post_ramp_pcc": best_pr_pcc if best_pr_ep is not None else None,
        "delta_from_gating": delta,
        "gating_changes_ckpt": gating_changes_ckpt,
        "status": status,
    })

# Summary
print("=" * 70)
print("SUMMARY")
print("=" * 70)
never_post_ramp = [r for r in results if r["best_post_ramp_epoch"] is None]
gating_helps    = [r for r in results
                   if r["gating_changes_ckpt"]
                   and r["delta_from_gating"] is not None
                   and r["delta_from_gating"] >= 0.005]
cosmetic        = [r for r in results
                   if r["gating_changes_ckpt"]
                   and r["delta_from_gating"] is not None
                   and abs(r["delta_from_gating"]) < 0.005]
already_ok      = [r for r in results
                   if not r["gating_changes_ckpt"]
                   and r["best_post_ramp_epoch"] is not None]

print(f"  Total MCSPR folds analysed : {len(results)}")
print(f"  Early-stopped pre-epoch 15 : {len(never_post_ramp)}  (MCSPR never fully active)")
print(f"  Ckpt already post-ramp     : {len(already_ok)}")
print(f"  Gating helps (Δ≥0.005)     : {len(gating_helps)}")
print(f"  Cosmetic change (Δ<0.005)  : {len(cosmetic)}")

if never_post_ramp:
    print("\n  Folds where MCSPR was never at full λ (re-run candidates):")
    for r in never_post_ramp:
        print(f"    {r['fold']}  ({r['total_epochs']} epochs total, "
              f"best ep {r['best_any_epoch']} pcc={r['best_any_pcc']:.4f})")

if gating_helps:
    print("\n  Folds where gating improves ckpt choice:")
    for r in gating_helps:
        print(f"    {r['fold']}: current ep {r['best_any_epoch']} "
              f"pcc={r['best_any_pcc']:.4f}  →  "
              f"post-ramp ep {r['best_post_ramp_epoch']} "
              f"pcc={r['best_post_ramp_pcc']:.4f}  Δ={r['delta_from_gating']:+.4f}")

if cosmetic:
    print("\n  Cosmetic (ep moves but PCC barely changes — keep current ckpt):")
    for r in cosmetic:
        print(f"    {r['fold']}: Δ={r['delta_from_gating']:+.4f}")

Path("logs").mkdir(exist_ok=True)
with open("logs/replay_best_epoch.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nFull results saved to logs/replay_best_epoch.json")
