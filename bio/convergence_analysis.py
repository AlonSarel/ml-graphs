import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

OUT_DIR = "/home/yandex/MLWG2025/alonsarel/updated_organized_results/figures/convergence_rates"

# Example list of pickle paths
pkl_paths = [
    "/home/yandex/MLWG2025/alonsarel/updated_organized_results/ours/60_emb/5_head/full/default_seed/our_finetune_supervised_masking_60_emb_5_head_100_epoch.pkl",
    "/home/yandex/MLWG2025/alonsarel/updated_organized_results/ours/60_emb/3_head/full/default_seed/second_run/our_full_60_emb_3_head_second_run.pkl",
    "/home/yandex/MLWG2025/alonsarel/updated_organized_results/original/300/full/default_seed/finetune_masking_then_supervised.pkl",
    "/home/yandex/MLWG2025/alonsarel/updated_organized_results/original/60/full/default_seed/run_original_full_60_emb_finetune.pkl",
    "/home/yandex/MLWG2025/alonsarel/updated_organized_results/ours/60_emb/5_head/full/seed_2/our_full_60_emb_5_head_seed_2.pkl",
    "/home/yandex/MLWG2025/alonsarel/updated_organized_results/ours/60_emb/3_head/from_supervised/deafult_seed/our_60_emb_from_supervised.pkl",
]

splits = ['train', 'val', 'test_easy', 'test_hard']
all_results = {split: [] for split in splits}
run_labels = []


# Load each pickle and collect average per epoch
for path in pkl_paths:
    run_name = os.path.splitext(os.path.basename(path))[0]  # file name without extension
    run_labels.append(run_name)
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    for split in splits:
        arr = np.array(data[split])       # shape: (epochs, tasks)
        mean_acc = arr.mean(axis=1)       # average over tasks
        all_results[split].append(mean_acc)

# Plot and save one file per split
epochs = range(len(all_results['train'][0]))

for split in splits:
    plt.figure(figsize=(8, 5))
    
    # Plot all runs for this split
    for curve, label in zip(all_results[split], run_labels):
        plt.plot(epochs, curve, label=label)
    
    plt.xlabel("Epoch")
    plt.ylabel("Average Accuracy")
    plt.title(f"{split} - Average Accuracy per Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(f"{OUT_DIR}/{split}_accuracy.png", dpi=300)
    plt.close()
