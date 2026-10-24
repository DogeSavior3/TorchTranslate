import torch
from torch.nn.utils.rnn import pad_sequence


def padding_func(batch, pad_ind, device):
    source_batch, target_batch = zip(*batch)

    sources = []
    targets = []
    source_length = []
    target_length = []

    for source, target in zip(source_batch, target_batch):
        source_length.append(len(source))
        target_length.append(len(target))

        sources.append(source.to(device))
        targets.append(target.to(device))

    source_padded = pad_sequence(sources, padding_value=pad_ind, batch_first=True).to(
        device
    )
    target_padded = pad_sequence(targets, padding_value=pad_ind, batch_first=True).to(
        device
    )
    source_lengths = torch.tensor(source_length, device=device)
    target_lengths = torch.tensor(target_length, device=device)

    return source_padded, target_padded, source_lengths, target_lengths
