"""
Plot training curves for a fine-tuning run: train/val loss and val nDCG@10.

Reads trainer_state.json from the run's latest checkpoint (the full log
history and the best checkpoint) and the zero-shot scores from summary.json,
so it works for any run made with a sentence-transformers Trainer.

Usage:
    python -m scripts.plot_training --model sparse --run run2
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display on the server
import matplotlib.pyplot as plt


def load_trainer_state(run_dir):
    """trainer_state.json from the latest checkpoint-N directory."""
    checkpoints = sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.rsplit("-", 1)[1]))
    if not checkpoints:
        raise SystemExit(f"No checkpoint-* directories in {run_dir}")
    return json.loads((checkpoints[-1] / "trainer_state.json").read_text())


def find_ndcg_key(history):
    """The evaluator's nDCG@10 key, e.g. eval_val_dot_ndcg@10 or eval_validation_cosine_ndcg@10."""
    for entry in history:
        for key in entry:
            if key.startswith("eval_") and key.endswith("ndcg@10"):
                return key
    raise SystemExit("No eval_*ndcg@10 entries in the log history")


def main():
    parser = argparse.ArgumentParser(description="Plot loss and validation nDCG@10 for a fine-tuning run.")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., sparse)")
    parser.add_argument("--run", type=str, required=True, help="Run name (e.g., run2)")
    args = parser.parse_args()

    run_dir = Path("checkpoints") / args.model / args.run
    title = f"{args.model.capitalize()} fine-tuning"
    
    # Save output to paper/figures/ at the repo root
    output = Path("paper") / "figures" / f"training_{args.model}_{args.run}.png"

    history = load_trainer_state(run_dir)["log_history"]
    state = load_trainer_state(run_dir)
    summary_path = run_dir / "summary.json"
    zero_shot = json.loads(summary_path.read_text()).get("zero_shot", {}) if summary_path.exists() else {}
    ndcg_key = find_ndcg_key(history)

    def points(key):
        """(epochs, values) for a logged key, with the zero-shot score at epoch 0 if saved."""
        pts = [(0.0, zero_shot[key])] if key in zero_shot else []
        pts += [(entry.get("epoch", 0.0), entry[key]) for entry in history if key in entry]
        return [e for e, _ in pts], [v for _, v in pts]

    # Best checkpoint (picked by nDCG@10 via metric_for_best_model)
    best_step = int(state["best_model_checkpoint"].rsplit("-", 1)[1])
    best_epoch = next(e["epoch"] for e in history if e.get("step") == best_step and ndcg_key in e)

    fig, (ax_loss, ax_ndcg) = plt.subplots(2, 1, sharex=True, figsize=(6, 6))

    ax_loss.plot(*points("loss"), label="Training loss", color="tab:blue", alpha=0.8)
    ax_loss.plot(*points("eval_loss"), label="Validation loss", color="tab:orange", marker="o", markersize=3)
    ax_loss.set(ylabel="Loss", title=title)

    epochs, ndcg = points(ndcg_key)
    ax_ndcg.plot(epochs, ndcg, label="Validation nDCG@10", color="tab:green", marker="o", markersize=3)
    ax_ndcg.set(xlabel="Epoch", ylabel="nDCG@10")

    for ax in (ax_loss, ax_ndcg):
        ax.axvline(best_epoch, color="gray", linestyle="--",
                   label=f"Best checkpoint (step {best_step}, epoch {best_epoch:.2f})")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    print(f"Saved {output} (best {ndcg_key} = {state['best_metric']:.4f} at step {best_step})")


if __name__ == "__main__":
    main()