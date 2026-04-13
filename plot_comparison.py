import torch
import matplotlib.pyplot as plt
import os
import numpy as np


def load_metrics(path):
    if os.path.exists(path):
        try:
            return torch.load(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None
    return None


def plot_single_metric(results_dirs, labels, metric_key, ylabel, title, save_path):
    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    found_data = False

    for i, (dir_path, label) in enumerate(zip(results_dirs, labels)):
        metric_path = os.path.join(dir_path, 'metrics.pth')
        metrics = load_metrics(metric_path)

        if metrics and 'test_episodes' in metrics and metric_key in metrics:
            episodes = np.array(metrics['test_episodes'])
            values = np.array(metrics[metric_key])

            # Check for length mismatch (in case training was interrupted)
            min_len = min(len(episodes), len(values))
            episodes = episodes[:min_len]
            values = values[:min_len]

            plt.plot(episodes, values, label=label, linewidth=2, color=colors[i % len(colors)])
            plt.scatter(episodes, values, s=20, color=colors[i % len(colors)])

            found_data = True
            print(f"[{title}] Loaded: {label} (Max: {np.max(values):.2f})")
        else:
            print(f"Warning: Could not load '{metric_key}' from {metric_path}")

    if found_data:
        plt.xlabel('Episodes', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot to {save_path}")
    else:
        print(f"No data found for {title}.")


if __name__ == "__main__":
    dirs = [
        'results/UAV-v0_default',  # World Model
        'results/UAV-v0_SAC_default',  # SAC
        'results/UAV-v0_DDPG_default',  # DDPG
    ]

    labels = [
        'World Model (Ours)',
        'SAC (Continuous)',
        'DDPG (Continuous)',
    ]

    # Reach Count
    plot_single_metric(
        dirs, labels,
        metric_key='test_rewards',  # 'test_rewards' stores Reach Count in your logic
        ylabel='Average Reach Count',
        title='Performance Comparison: Reach Count',
        save_path='comparison_reach_count.png'
    )

    # Average Reward
    plot_single_metric(
        dirs, labels,
        metric_key='test_avg_rewards',  # New key we added
        ylabel='Avg. Accumulated Reward (per 5000 steps)',
        title='Performance Comparison: Average Reward',
        save_path='comparison_avg_reward.png'
    )