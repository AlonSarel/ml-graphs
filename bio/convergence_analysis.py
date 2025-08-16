import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# TODO: Remove full paths from git...

OUT_DIR = "/home/yandex/MLWG2025/alonsarel/updated_organized_results/figures/convergence_rates"

# Define groups: label -> list of pickle paths
groups = {
    "Original Model (300 emb-size)": [
        "/home/yandex/MLWG2025/alonsarel/updated_organized_results/original/300/full/default_seed/finetune_masking_then_supervised.pkl",
        "/home/yandex/MLWG2025/alonsarel/updated_organized_results/original/300/full/seed_2/run_original_full_300_emb_seed_set_2.pkl",
    ],
    "Original Model (60 emb-size)": [
        "/home/yandex/MLWG2025/alonsarel/updated_organized_results/original/60/full/default_seed/run_original_full_60_emb_finetune.pkl",
        "/home/yandex/MLWG2025/alonsarel/updated_organized_results/original/60/full/seed_1/run_original_full_60_emb_seed_set_1.pkl",
        "/home/yandex/MLWG2025/alonsarel/updated_organized_results/original/60/full/seed_2/run_original_full_60_emb_seed_set_2.pkl",
    ],
    "Multi-Head Model (60 emb-size, 5 heads)": [
        "/home/yandex/MLWG2025/alonsarel/updated_organized_results/ours/60_emb/5_head/full/seed_2/our_full_60_emb_5_head_seed_2.pkl",
        "/home/yandex/MLWG2025/alonsarel/updated_organized_results/ours/60_emb/5_head/full/default_seed/our_finetune_supervised_masking_60_emb_5_head_100_epoch.pkl",
    ],
    "Multi-Head Model (60 emb-size, 3 heads)": [
        "/home/yandex/MLWG2025/alonsarel/updated_organized_results/ours/60_emb/3_head/full/default_seed/second_run/our_full_60_emb_3_head_second_run.pkl",
        "/home/yandex/MLWG2025/alonsarel/updated_organized_results/ours/60_emb/3_head/full/seed_1/our_full_60_emb_3_head_seed_1.pkl",
        "/home/yandex/MLWG2025/alonsarel/updated_organized_results/ours/60_emb/3_head/full/seed_2/our_full_60_emb_3_head_seed_2.pkl",
    ]
}

splits = ['train', 'val', 'test_easy', 'test_hard']

def load_and_average(pkl_paths, splits):
    """Load runs, compute per-epoch average over tasks, then average over runs."""
    per_run_curves = {split: [] for split in splits}
    for path in pkl_paths:
        with open(path, "rb") as f:
            data = pickle.load(f)
        for split in splits:
            arr = np.array(data[split])    # shape: (epochs, tasks)
            mean_acc = arr.mean(axis=1)    # average over tasks
            per_run_curves[split].append(mean_acc)
    
    # average across runs
    avg_curves = {split: np.mean(per_run_curves[split], axis=0) for split in splits}
    return avg_curves

# Compute averages for all groups
group_results = {name: load_and_average(paths, splits) for name, paths in groups.items()}

# Epochs
epochs = range(len(next(iter(group_results.values()))['train']))
os.makedirs(OUT_DIR, exist_ok=True)

# === Plot Train + Validation (same color, val dashed) ===
plt.figure(figsize=(8, 5))

colors = ["red", "green", "blue", "orange"]

for (name, results), color in zip(group_results.items(), colors):
    # Train curve (solid)
    plt.plot(epochs, results['train'], label=f"{name}", linewidth=2, color=color)
    # Val curve (dashed, same color)
    plt.plot(epochs, results['val'], linestyle="--", linewidth=2, color=color)

plt.xlabel("Epoch", fontsize=16)
plt.ylabel("ROC-AUC Score", fontsize=16)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(f"{OUT_DIR}/train_vs_val_accuracy.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))

for (name, results), color in zip(group_results.items(), colors):
    plt.plot(epochs, results['test_hard'], label=f"{name}", linewidth=2, color=color)

plt.xlabel("Epoch", fontsize=16)
plt.ylabel("ROC-AUC Score", fontsize=16)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(f"{OUT_DIR}/test_hard_accuracy.png", dpi=300)
plt.close()