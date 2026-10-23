import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from twhin_embedding_cache import make_cached_embedder  # noqa: E402


def recording_embed(calls):
    def embed(_model, _tokenizer, texts, _batch_size):
        calls.append(list(texts))
        return torch.tensor([[float(len(t)), float(sum(map(ord, t)))] for t in texts])

    return embed


def test_matches_uncached_output_in_small_batches():
    calls = []
    embed = make_cached_embedder(recording_embed(calls), chunk_size=2)
    texts = ["a", "bb", "ccc", "a", "dddd"]

    result = embed(object(), None, texts, batch_size=1000)

    assert torch.equal(result, recording_embed([])(None, None, texts, 1000))
    assert all(len(chunk) <= 2 for chunk in calls)
    assert sum(len(chunk) for chunk in calls) == 4  # the duplicate "a" is embedded once


def test_later_rounds_only_embed_new_texts():
    calls = []
    embed = make_cached_embedder(recording_embed(calls), chunk_size=8)
    model = object()
    embed(model, None, ["profile v1", "post 1"], batch_size=1000)
    calls.clear()

    result = embed(model, None, ["profile v2", "post 1", "post 2"], batch_size=1000)

    assert calls == [["profile v2", "post 2"]]
    assert result.shape == (3, 2)
