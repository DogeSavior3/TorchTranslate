from functools import partial
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.datasets.collate import padding_func
from src.trainer.training import train
from src.utils.setup import build_vocab, get_device


@hydra.main(version_base=None, config_path="src/configs", config_name="transformer")
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    device = get_device(cfg.device)
    source_vocab = build_vocab(cfg.data.train_source, cfg.data.min_freq)
    target_vocab = build_vocab(cfg.data.train_target, cfg.data.min_freq)
    collate = partial(padding_func, pad_ind=source_vocab.pad_ind, device=device)
    loaders = []
    for split in ("train", "val"):
        dataset = instantiate(
            cfg.dataset,
            source_file=to_absolute_path(cfg.data[f"{split}_source"]),
            target_file=to_absolute_path(cfg.data[f"{split}_target"]),
            source_vocab=source_vocab,
            target_vocab=target_vocab,
        )
        loaders.append(
            DataLoader(
                dataset,
                batch_size=cfg.data.batch_size,
                shuffle=split == "train",
                collate_fn=collate,
            )
        )
    model_config = OmegaConf.merge(
        cfg.model,
        {
            "source_vocab_size": len(source_vocab),
            "target_vocab_size": len(target_vocab),
        },
    )
    model = instantiate(model_config, device=device)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler_kwargs = (
        {"T_max": cfg.trainer.epochs * len(loaders[0])}
        if "T_max" in list(cfg.scheduler)
        else {}
    )
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer, **scheduler_kwargs)
    output_dir = Path(to_absolute_path(cfg.output.dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    train(
        model,
        optimizer,
        scheduler,
        loaders[0],
        loaders[1],
        cfg.trainer.epochs,
        device,
        source_vocab.pad_ind,
        source_vocab,
        target_vocab,
        save_path=output_dir / cfg.output.checkpoint,
        label_smoothing=cfg.trainer.label_smoothing,
        grad_clip=cfg.trainer.grad_clip,
        bleu_kwargs={
            "val_de_path": to_absolute_path(cfg.data.val_source),
            "val_en_path": to_absolute_path(cfg.data.val_target),
            "stride": cfg.evaluation.stride,
            "max_len": cfg.decoding.max_len,
            "beam_size": cfg.evaluation.beam_size,
        },
        show=cfg.plotting.show,
        plot_path=output_dir / cfg.plotting.filename if cfg.plotting.save else None,
        checkpoint_metadata={
            "model_config": OmegaConf.to_container(model_config, resolve=True),
            "source_vocab": vars(source_vocab),
            "target_vocab": vars(target_vocab),
        },
    )


if __name__ == "__main__":
    main()
