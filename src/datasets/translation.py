import torch
from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    def __init__(
        self,
        source_file,
        target_file,
        source_vocab=None,
        target_vocab=None,
        device="cpu",
    ):
        super().__init__()
        self.device = device
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

        with open(source_file) as f:
            self.source_sentences = [line.strip() for line in f]

        with open(target_file) as f:
            self.target_sentences = [line.strip() for line in f]

    def __len__(self):
        return len(self.source_sentences)

    def __getitem__(self, index):
        source_sentence = self.source_sentences[index]
        target_sentence = self.target_sentences[index]

        source_indices = self.source_vocab.encode(source_sentence)
        target_indices = self.target_vocab.encode(target_sentence)

        return torch.tensor(
            source_indices, dtype=torch.long, device=self.device
        ), torch.tensor(target_indices, dtype=torch.long, device=self.device)
