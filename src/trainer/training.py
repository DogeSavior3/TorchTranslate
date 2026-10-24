import torch
from torch import nn
from tqdm.auto import tqdm

from src.metrics.bleu import compute_val_bleu
from src.trainer.epochs import train_epoch, val_epoch
from src.utils.plotting import plot_losses


def train(
    model,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    num_epochs,
    device,
    pad_ind,
    source_vocab,
    target_vocab,
    save_path="./best_model.pt",
    label_smoothing=0.1,
    grad_clip=1.0,
    bleu_kwargs=None,
    show=True,
    plot_path=None,
    checkpoint_metadata=None,
):
    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_ind, label_smoothing=label_smoothing
    ).to(device)
    train_losses = []
    val_losses = []
    val_bleus = []
    best_bleu = -float("inf")
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    for epochs in tqdm(range(1, num_epochs + 1), desc="Epoch"):
        train_loss = train_epoch(
            model,
            optimizer,
            scheduler,
            criterion,
            train_loader,
            device,
            pad_ind,
            scaler,
            grad_clip,
        )
        train_losses.append(train_loss)
        val_loss = val_epoch(model, optimizer, criterion, val_loader, device, pad_ind)
        val_losses.append(val_loss)
        if epochs > 0:
            val_bleu = compute_val_bleu(
                model, source_vocab, target_vocab, device, **(bleu_kwargs or {})
            )
            val_bleus.append(val_bleu)

            if val_bleu > best_bleu:
                best_bleu = val_bleu
                torch.save(
                    {
                        **(checkpoint_metadata or {}),
                        "epoch": epochs,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_bleu": val_bleu,
                    },
                    save_path,
                )

        plot_losses(train_losses, val_losses, val_bleus, show=show, save_path=plot_path)

    return train_losses, val_losses, val_bleus
