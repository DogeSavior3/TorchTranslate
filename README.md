# TorchTranslate

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This project implements German-to-English translation with PyTorch's `nn.Transformer`. It uses word-level vocabularies, sinusoidal positional encodings, beam search decoding, and SacreBLEU evaluation. Training and inference are configured with Hydra.

## Installation

The project uses [uv](https://docs.astral.sh/uv/) to manage Python and dependencies.

1. Install uv using the [official instructions](https://docs.astral.sh/uv/getting-started/installation/). On macOS, you can use Homebrew:

   ```bash
   brew install uv
   ```

2. Clone the repository and install the locked dependencies:

   ```bash
   git clone https://github.com/DogeSavior3/bhw2_dl.git
   cd bhw2_dl
   uv sync --locked --python 3.13
   ```

   uv creates a `.venv` environment. Dependencies are defined in `pyproject.toml` and pinned in `uv.lock`. `requirements.txt` provides an alternative pip-compatible export.

## How To Use

To train a model, run:

```bash
uv run python train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from [src/configs](src/configs) and `HYDRA_CONFIG_ARGUMENTS` are optional overrides. The default training config is `transformer`. To use the model and training settings recovered from the project report:

```bash
uv run python train.py -cn=final_config
```

To translate a sentence, translate a file, or evaluate validation BLEU:

```bash
uv run python inference.py HYDRA_CONFIG_ARGUMENTS
```

Set `sentence='german text'` for a single sentence, `input_file=PATH output_file=PATH` for a file, or `evaluate=true` for BLEU evaluation. The default checkpoint is `output/best_model.pt`.

Parallel training and validation texts are stored in `data/`. German is the source language and English is the target language. Vocabularies are built from the training split. Checkpoints include the vocabulary mappings and model configuration needed for inference.

Training saves the checkpoint with the best validation BLEU and a loss/BLEU plot to `output/`. Plot display and saving are controlled by `plotting.show` and `plotting.save`. BLEU uses every third validation sentence by default; set `evaluation.stride=1` to evaluate the full split.

## Credits

Translation quality is evaluated with [SacreBLEU](https://github.com/mjpost/sacrebleu).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
