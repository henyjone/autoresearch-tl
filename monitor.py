"""
Real-time training monitor for autoresearch-tl.
Watches progress.log and displays live training metrics.

Usage:
    python monitor.py              # auto-refresh every 2 seconds
    python monitor.py --once       # print current status once and exit
    python monitor.py --plot       # show loss curve plot (requires matplotlib)
"""

import os
import sys
import time
import argparse

PROGRESS_LOG = os.path.join(os.path.dirname(__file__), "progress.log")


def read_progress():
    """Read progress.log and return list of dicts."""
    if not os.path.exists(PROGRESS_LOG):
        return []
    rows = []
    with open(PROGRESS_LOG, "r") as f:
        lines = f.readlines()
    if len(lines) < 2:
        return []
    headers = lines[0].strip().split("\t")
    for line in lines[1:]:
        parts = line.strip().split("\t")
        if len(parts) == len(headers):
            row = {}
            for h, v in zip(headers, parts):
                try:
                    row[h] = float(v)
                except ValueError:
                    row[h] = v
            rows.append(row)
    return rows


def print_status(rows):
    """Print formatted training status."""
    os.system("cls" if os.name == "nt" else "clear")
    print("=" * 70)
    print("  autoresearch-tl Training Monitor")
    print("=" * 70)

    if not rows:
        print("\n  Waiting for training to start...")
        print(f"  Watching: {PROGRESS_LOG}")
        return

    latest = rows[-1]
    progress = latest.get("progress", 0)
    step = int(latest.get("step", 0))
    loss = latest.get("loss", 0)
    rmse = latest.get("rmse_approx", 0)
    lr = latest.get("lr", 0)
    epoch = int(latest.get("epoch", 0))
    remaining = latest.get("remaining_s", 0)

    # Progress bar
    bar_width = 40
    filled = int(bar_width * progress / 100)
    bar = "█" * filled + "░" * (bar_width - filled)
    print(f"\n  Progress: [{bar}] {progress:.1f}%")
    print(f"  Remaining: {remaining:.0f}s")
    print()
    print(f"  Step:      {step:,}")
    print(f"  Loss:      {loss:.4f}")
    print(f"  ~RMSE:     {rmse:.2f} dB")
    print(f"  LR:        {lr:.6f}")
    print(f"  Epoch:     {epoch}")
    print()

    # Show improvement trend (last 10 entries)
    if len(rows) >= 2:
        print("  Recent trend:")
        print("  " + "-" * 55)
        print(f"  {'Step':>8}  {'Progress':>8}  {'Loss':>10}  {'~RMSE (dB)':>10}  {'LR':>10}")
        print("  " + "-" * 55)
        recent = rows[-10:]
        for r in recent:
            print(f"  {int(r.get('step',0)):>8,}  {r.get('progress',0):>7.1f}%  {r.get('loss',0):>10.4f}  {r.get('rmse_approx',0):>10.2f}  {r.get('lr',0):>10.6f}")

    # Best RMSE
    if rows:
        best = min(rows, key=lambda r: r.get("rmse_approx", float("inf")))
        print()
        print(f"  Best ~RMSE so far: {best.get('rmse_approx', 0):.2f} dB (step {int(best.get('step', 0)):,})")

    print()
    print("  Press Ctrl+C to stop monitoring")


def plot_curves(rows):
    """Plot training loss and RMSE curves."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")
        return

    if not rows:
        print("No data to plot.")
        return

    steps = [r["step"] for r in rows]
    losses = [r["loss"] for r in rows]
    rmses = [r["rmse_approx"] for r in rows]
    lrs = [r["lr"] for r in rows]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(steps, losses, "b-", linewidth=1)
    axes[0].set_ylabel("Loss (MSE)")
    axes[0].set_title("autoresearch-tl Training Progress")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, rmses, "r-", linewidth=1)
    axes[1].set_ylabel("~RMSE (dB)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, lrs, "g-", linewidth=1)
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_xlabel("Step")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "training_curve.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Monitor autoresearch-tl training")
    parser.add_argument("--once", action="store_true", help="Print status once and exit")
    parser.add_argument("--plot", action="store_true", help="Plot training curves")
    parser.add_argument("--interval", type=float, default=2.0, help="Refresh interval in seconds (default: 2)")
    args = parser.parse_args()

    if args.plot:
        rows = read_progress()
        plot_curves(rows)
        return

    if args.once:
        rows = read_progress()
        print_status(rows)
        return

    # Live monitoring
    try:
        while True:
            rows = read_progress()
            print_status(rows)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n  Monitor stopped.")


if __name__ == "__main__":
    main()
