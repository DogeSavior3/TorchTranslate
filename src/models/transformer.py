import math

import torch
from torch import nn

from src.models.positional_encoding import PositionalEncoding


class Transformer(nn.Module):
    def __init__(
        self,
        source_vocab_size,
        target_vocab_size,
        d_model,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        d_ff,
        dropout,
        max_len,
        device="cpu",
    ):
        super().__init__()
        self.source_emb = nn.Embedding(source_vocab_size, d_model, device=device)
        self.target_emb = nn.Embedding(target_vocab_size, d_model, device=device)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len, device)

        self.transformer = nn.Transformer(
            d_model,
            nhead,
            num_encoder_layers,
            num_decoder_layers,
            d_ff,
            dropout,
            batch_first=True,
            device=device,
        )

        self.fc_out = nn.Linear(d_model, target_vocab_size, device=device)
        self.d_model = d_model
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.source_emb.weight.data.uniform_(-initrange, initrange)
        self.target_emb.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def forward(self, source, target, source_pad_mask, target_pad_mask):
        source_emb = self.source_emb(source) * math.sqrt(self.d_model)
        target_emb = self.target_emb(target) * math.sqrt(self.d_model)

        source_emb = self.pos_encoder(source_emb)
        target_emb = self.pos_encoder(target_emb)

        target_mask = torch.triu(
            torch.ones(target.size(1), target.size(1)) * float("-inf"), diagonal=1
        ).to(source.device)

        out = self.transformer(
            source_emb,
            target_emb,
            tgt_mask=target_mask,
            src_key_padding_mask=source_pad_mask,
            tgt_key_padding_mask=target_pad_mask,
            memory_key_padding_mask=source_pad_mask,
        )

        return self.fc_out(out)
