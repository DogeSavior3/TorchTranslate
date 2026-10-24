from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig

from src.datasets.vocabulary import Vocabulary
from src.metrics.bleu import compute_val_bleu
from src.utils.setup import get_device
from src.utils.translation import translate


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(cfg: DictConfig):
    device = get_device(cfg.device)
    checkpoint = torch.load(
        to_absolute_path(cfg.checkpoint), map_location=device, weights_only=True
    )
    source_vocab, target_vocab = Vocabulary(), Vocabulary()
    source_vocab.__dict__.update(checkpoint["source_vocab"])
    target_vocab.__dict__.update(checkpoint["target_vocab"])
    model = instantiate(checkpoint["model_config"], device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if cfg.evaluate:
        print(
            compute_val_bleu(
                model,
                source_vocab,
                target_vocab,
                device,
                val_de_path=to_absolute_path(cfg.data.val_source),
                val_en_path=to_absolute_path(cfg.data.val_target),
                stride=cfg.evaluation.stride,
                beam_size=cfg.evaluation.beam_size,
                max_len=cfg.decoding.max_len,
            )
        )
    else:
        if cfg.input_file is not None:
            with open(to_absolute_path(cfg.input_file)) as f:
                sentences = [line.strip() for line in f]
        else:
            sentences = [cfg.sentence]
        translations = [
            translate(
                model,
                sentence,
                source_vocab,
                target_vocab,
                device,
                max_len=cfg.decoding.max_len,
                beam_size=cfg.decoding.beam_size,
            )
            for sentence in sentences
        ]
        if cfg.output_file is not None:
            path = Path(to_absolute_path(cfg.output_file))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(translations) + "\n")
        else:
            print("\n".join(translations))


if __name__ == "__main__":
    main()
