import torch
from hydra.utils import to_absolute_path

from src.datasets.vocabulary import Vocabulary


def get_device(name):
    return torch.device(
        ("cuda:0" if torch.cuda.is_available() else "cpu") if name == "auto" else name
    )


def build_vocab(path, min_freq):
    with open(to_absolute_path(path)) as f:
        sentences = [line.strip() for line in f]
    vocab = Vocabulary()
    vocab.build_vocab(sentences, min_freq=min_freq)
    return vocab
