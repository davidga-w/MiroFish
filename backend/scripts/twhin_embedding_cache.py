"""
Bounded-memory TwHIN-BERT embeddings for the OASIS Twitter recommender.

Every round, OASIS's ``rec_sys_personalized_twh`` re-embeds all user profiles
and all posts in one batch of up to 1000 texts. BERT attention memory grows
with batch size x sequence length squared, so long runs get OOM-killed once a
few hundred posts exist.

``install_twhin_embedding_cache()`` replaces that embedding call with one that
embeds only texts it has not seen yet, in small batches. Each text still gets
the same vector, so recommendations are unchanged; peak memory stays flat and
each round only pays for new posts and edited profiles.
"""

import torch

DEFAULT_CHUNK_SIZE = 16


def make_cached_embedder(embed, chunk_size=DEFAULT_CHUNK_SIZE):
    """Wrap ``embed(model, tokenizer, texts, batch_size)`` with a per-text cache."""
    cache = {}

    def cached_embed(model, tokenizer, texts, batch_size=None):
        # OASIS passes batch_size=1000; chunk_size is what bounds memory.
        keys = [(id(model), text) for text in texts]
        missing = [key for key in dict.fromkeys(keys) if key not in cache]
        for start in range(0, len(missing), chunk_size):
            chunk = missing[start:start + chunk_size]
            vectors = embed(model, tokenizer, [text for _, text in chunk], chunk_size)
            cache.update(zip(chunk, vectors))
        result = torch.stack([cache[key] for key in keys])
        # Drop texts the current corpus no longer uses (superseded profiles).
        for key in set(cache) - set(keys):
            del cache[key]
        return result

    return cached_embed


def install_twhin_embedding_cache():
    from oasis.social_platform import recsys

    recsys.generate_post_vector = make_cached_embedder(recsys.generate_post_vector)
