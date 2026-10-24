import torch
from torch import nn
from tqdm.auto import tqdm


def train_epoch(
    model,
    optimizer,
    scheduler,
    criterion,
    loader,
    device,
    pad_ind,
    scaler,
    grad_clip=1.0,
):
    model.train()
    epoch_loss = 0.0

    for source, target, source_len, target_len in tqdm(
        loader, desc="train", leave=True
    ):
        source = source.to(device)
        target = target.to(device)
        source_len = source_len.to(device)
        target_len = target_len.to(device)

        target_input = target[:, :-1]
        target_output = target[:, 1:]

        source_pad_mask = (source == pad_ind).to(device)
        target_pad_mask = (target_input == pad_ind).to(device)
        optimizer.zero_grad()

        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits = model(source, target_input, source_pad_mask, target_pad_mask)
            logits_flat = logits.reshape(-1, logits.size(-1))
            target_flat = target_output.reshape(-1)
            loss = criterion(logits_flat, target_flat)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        epoch_loss += loss.item() * source.size(0)

    return epoch_loss / len(loader.dataset)


@torch.no_grad()
def val_epoch(model, optimizer, criterion, loader, device, pad_ind):
    model.eval()
    epoch_loss = 0.0

    for source, target, source_len, target_len in tqdm(loader, desc="val", leave=True):
        source = source.to(device)
        target = target.to(device)
        source_len = source_len.to(device)
        target_len = target_len.to(device)

        target_input = target[:, :-1]
        target_output = target[:, 1:]
        source_pad_mask = (source == pad_ind).to(device)
        target_pad_mask = (target_input == pad_ind).to(device)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits = model(source, target_input, source_pad_mask, target_pad_mask)

            logits_flat = logits.reshape(-1, logits.size(-1))
            target_flat = target_output.reshape(-1)
            loss = criterion(logits_flat, target_flat)

        epoch_loss += loss.item() * source.size(0)

    return epoch_loss / len(loader.dataset)
