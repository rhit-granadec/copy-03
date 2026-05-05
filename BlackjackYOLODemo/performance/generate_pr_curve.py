"""
performance/generate_pr_curve.py — Generate Precision-Recall curve from bot run data.

Reads pr_raw.jsonl (written by app.py during bot runs) and sweeps
confidence thresholds to produce a P-R curve + optimal F1 operating point.

Usage:
    python performance/generate_pr_curve.py
"""

import json
import os
import sys

import numpy as np

PR_LOG = os.path.join(os.path.dirname(__file__), "pr_raw.jsonl")


def load_hands(path: str) -> list[dict]:
    """Load all hand records from the JSONL log."""
    hands = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                hands.append(json.loads(line))
    return hands


def evaluate_at_threshold(hands: list[dict], threshold: float):
    """Compute TP, FP, FN across all hands at a given confidence threshold."""
    tp = fp = fn = 0
    for hand in hands:
        actual = list(hand["actual"])
        # Filter raw detections at this threshold
        dets = [d for d in hand["raw_detections"] if d["confidence"] >= threshold]

        # Match detections to actual cards (greedy, same logic as app.py)
        det_pool = [d["class_name"] for d in dets]
        for card in actual:
            idx = None
            for i, d in enumerate(det_pool):
                if d == card:
                    idx = i
                    break
            if idx is not None:
                tp += 1
                det_pool.pop(idx)
            else:
                fn += 1
        fp += len(det_pool)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall, tp, fp, fn


def main():
    if not os.path.exists(PR_LOG):
        print(f"No data found at {PR_LOG}")
        print("Run the bot for several hands first, then re-run this script.")
        sys.exit(1)

    hands = load_hands(PR_LOG)
    print(f"Loaded {len(hands)} hands from {PR_LOG}")

    total_actual = sum(len(h["actual"]) for h in hands)
    total_raw = sum(len(h["raw_detections"]) for h in hands)
    print(f"Total actual cards: {total_actual}, total raw detections: {total_raw}")

    # Sweep thresholds
    thresholds = np.arange(0.05, 0.96, 0.01)
    precisions = []
    recalls = []
    f1s = []

    print(f"\n{'Thresh':>7} {'Prec':>7} {'Recall':>7} {'F1':>7} {'TP':>5} {'FP':>5} {'FN':>5}")
    print("-" * 52)

    for t in thresholds:
        p, r, tp, fp, fn = evaluate_at_threshold(hands, t)
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        if int(t * 100) % 5 == 0:
            print(f"{t:>7.2f} {p:>7.1%} {r:>7.1%} {f1:>7.3f} {tp:>5} {fp:>5} {fn:>5}")

    # Find optimal F1
    best_idx = int(np.argmax(f1s))
    best_t = thresholds[best_idx]
    best_p = precisions[best_idx]
    best_r = recalls[best_idx]
    best_f1 = f1s[best_idx]

    print(f"\nOptimal threshold: {best_t:.2f}")
    print(f"  Precision: {best_p:.1%}")
    print(f"  Recall:    {best_r:.1%}")
    print(f"  F1:        {best_f1:.3f}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # P-R curve
        ax1.plot(recalls, precisions, "b-", linewidth=2)
        ax1.plot(best_r, best_p, "r*", markersize=15,
                 label=f"Best F1={best_f1:.3f} @ t={best_t:.2f}")
        ax1.set_xlabel("Recall", fontsize=12)
        ax1.set_ylabel("Precision", fontsize=12)
        ax1.set_title("Precision-Recall Curve", fontsize=14)
        ax1.set_xlim(0, 1.05)
        ax1.set_ylim(0, 1.05)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)

        # F1 / Precision / Recall vs threshold
        ax2.plot(thresholds, f1s, "g-", linewidth=2, label="F1")
        ax2.plot(thresholds, precisions, "b--", linewidth=1.5, label="Precision")
        ax2.plot(thresholds, recalls, "r--", linewidth=1.5, label="Recall")
        ax2.axvline(best_t, color="gray", linestyle=":", alpha=0.7,
                     label=f"Optimal t={best_t:.2f}")
        ax2.set_xlabel("Confidence Threshold", fontsize=12)
        ax2.set_ylabel("Score", fontsize=12)
        ax2.set_title("Metrics vs Confidence Threshold", fontsize=14)
        ax2.set_xlim(0.05, 0.95)
        ax2.set_ylim(0, 1.05)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)

        plt.tight_layout()
        out_path = os.path.join(os.path.dirname(__file__), "pr_curve.png")
        plt.savefig(out_path, dpi=150)
        print(f"\nPlot saved to {out_path}")

    except ImportError:
        print("\nmatplotlib not installed — skipping plot generation.")
        print("Install with: pip install matplotlib")


if __name__ == "__main__":
    main()
