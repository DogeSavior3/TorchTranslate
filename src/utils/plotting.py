from pathlib import Path

import matplotlib.pyplot as plt


def plot_losses(train_losses, val_losses, val_bleus, show=True, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(train_losses) + 1)

    axes[0].plot(epochs, train_losses, label="train")
    axes[0].plot(epochs, val_losses, label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].grid(True)
    axes[0].legend()
    if len(val_bleus):
        axes[1].plot(range(1, len(val_bleus) + 1), val_bleus, label="val BLEU")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("BLEU")
        axes[1].grid(True)
        axes[1].legend()

    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
