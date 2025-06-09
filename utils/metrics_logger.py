import os
import pandas as pd
import matplotlib.pyplot as plt

def save_metrics_to_excel(metrics: dict, file_path: str):
    # 自动裁剪不等长字段
    lengths = [len(v) for v in metrics.values() if isinstance(v, list)]
    min_len = min(lengths)

    # 截断成相同长度
    clean_metrics = {k: v[:min_len] for k, v in metrics.items() if isinstance(v, list)}

    df = pd.DataFrame(clean_metrics)

    # 字段重命名
    col_map = {
        'round': 'Round',
        'epoch': 'Epoch',
        'iteration': 'Round',
        'accuracy': 'Accuracy',
        'loss': 'Loss',
        'backdoor_success_rate': 'Backdoor SR'
    }
    df.rename(columns=col_map, inplace=True)

    df.to_excel(file_path, index=False)
    print(f"Excel saved at: {file_path}")

def plot_metrics(metrics: dict, save_dir: str, title_prefix: str):
    import matplotlib.pyplot as plt
    import os

    # 自动裁剪不等长字段
    lengths = [len(v) for v in metrics.values() if isinstance(v, list)]
    min_len = min(lengths)

    # 截断为统一长度
    metrics = {k: v[:min_len] for k, v in metrics.items() if isinstance(v, list)}

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    if 'round' in metrics:
        x_label = 'Round'; x = metrics['round']
    elif 'epoch' in metrics:
        x_label = 'Epoch'; x = metrics['epoch']
    elif 'iteration' in metrics:
        x_label = 'Round'; x = metrics['iteration']
    else:
        x_label = 'Step'; x = list(range(1, len(next(iter(metrics.values()))) + 1))

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx() if 'loss' in metrics else None
    ax3 = ax1.twinx() if 'backdoor_success_rate' in metrics else None

    if ax3 and ax2:
        ax3.spines['right'].set_position(('outward', 60))

    if 'accuracy' in metrics:
        ax1.plot(x, metrics['accuracy'], color='blue', marker='o', label='Accuracy')
        ax1.set_ylabel('Accuracy (%)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_ylim(0, 100)

    if 'loss' in metrics and ax2:
        ax2.plot(x, metrics['loss'], color='red', marker='s', label='Loss')
        ax2.set_ylabel('Loss', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

    if 'backdoor_success_rate' in metrics and ax3:
        ax3.plot(x, metrics['backdoor_success_rate'], color='green', marker='^', label='Backdoor SR')
        ax3.set_ylabel('Backdoor SR (%)', color='green')
        ax3.tick_params(axis='y', labelcolor='green')
        ax3.set_ylim(0, 100)
    ax1.set_xlabel(x_label)
    plt.title(f"{title_prefix} Metrics")
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f"{title_prefix.replace(' ', '_').lower()}_metrics.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Plots saved at: {plot_path}")