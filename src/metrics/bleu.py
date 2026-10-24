import os
import subprocess

from tqdm.auto import tqdm

from src.utils.translation import translate


def compute_val_bleu(
    model,
    source_vocab,
    target_vocab,
    device,
    val_de_path="data/val.de-en.de",
    val_en_path="data/val.de-en.en",
    stride=3,
    max_len=82,
    beam_size=1,
):
    model.eval()
    temp_pred_path1 = "temp_val1.en"
    temp_pred_path2 = "temp_val2.en"
    with open(val_de_path, "r", encoding="utf-8") as f:
        val_sentences = [line.strip() for line in f]

    translations = []
    for sent in tqdm(val_sentences[::stride], desc="Translating val", leave=False):
        en_translation = translate(
            model,
            sent,
            source_vocab,
            target_vocab,
            device,
            beam_size=beam_size,
            max_len=max_len,
        )
        translations.append(en_translation)

    with open(temp_pred_path1, "w", encoding="utf-8") as f:
        f.writelines(translation + "\n" for translation in translations)

    with open(val_en_path, "r", encoding="utf-8") as f:
        val_en_sentences = [line.strip() for line in f]

    with open(temp_pred_path2, "w", encoding="utf-8") as f:
        for sent in val_en_sentences[::stride]:
            f.write(sent + "\n")

    with open(temp_pred_path1, "r", encoding="utf-8") as pred_file:
        result = subprocess.run(
            ["sacrebleu", temp_pred_path2, "--tokenize", "none", "--width", "2", "-b"],
            stdin=pred_file,
            capture_output=True,
            text=True,
            check=True,
        )

    bleu = float(result.stdout.strip())
    os.remove(temp_pred_path1)
    os.remove(temp_pred_path2)
    return bleu
