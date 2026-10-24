import torch

from src.datasets.vocabulary import Vocabulary
from src.models.transformer import Transformer


@torch.no_grad()
def translate(
    model: Transformer,
    source_sentence,
    source_vocab,
    target_vocab: Vocabulary,
    device,
    max_len=82,
    beam_size=3,
):
    model.eval()

    source_tokens = source_vocab.encode(source_sentence)
    source_tensor = torch.tensor(source_tokens).unsqueeze(0).to(device)

    beams = [([target_vocab.bos_ind], 0.0)]

    forbidden_tokens = [
        target_vocab.unk_ind,
        target_vocab.pad_ind,
        target_vocab.bos_ind,
    ]

    def sort_func(x):
        return x[1]

    for _ in range(max_len):
        all_candidates = []

        for translation, score in beams:
            if translation[-1] == target_vocab.eos_ind:
                all_candidates.append((translation, score))
                continue

            target_tensor = torch.tensor(translation).unsqueeze(0).to(device)

            logits = model(source_tensor, target_tensor, None, None)
            next_logits = logits[0, -1, :].clone()
            next_logits[forbidden_tokens] = -float("inf")
            if len(translation) >= 2 and translation[-1] == translation[-2]:
                next_logits[translation[-1]] = -float("inf")

            log_probs = torch.log_softmax(next_logits, dim=-1)
            topk_log_probs, topk_indices = torch.topk(log_probs, beam_size)

            for log_prob, token_id in zip(
                topk_log_probs.tolist(), topk_indices.tolist()
            ):
                new_translation = translation + [token_id]
                new_score = score + log_prob
                all_candidates.append((new_translation, new_score))

        all_candidates.sort(key=sort_func, reverse=True)
        beams = all_candidates[:beam_size]

        if all(translation[-1] == target_vocab.eos_ind for translation, _ in beams):
            break

    best_translation, _ = max(beams, key=sort_func)

    result = best_translation[1:]
    if target_vocab.eos_ind in result:
        result = result[: result.index(target_vocab.eos_ind)]

    return target_vocab.decode(result)
