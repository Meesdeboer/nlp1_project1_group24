import json

import numpy as np
import torch
from huggingface_hub import hf_hub_download

# GloVe 6B (Wikipedia 2014 + Gigaword 5, 300 dimensions, lowercased; Pennington, Socher & Manning, 2014),
# as packaged by sentence-transformers. Downloaded once (about 480 MB) into the Hugging Face cache.
REPO = "sentence-transformers/average_word_embeddings_glove.6B.300d"


def load_glove():
    """Return (word_to_row, matrix): a dict word -> row number and the float32 embedding matrix [400001, 300]."""
    vocab = json.load(open(hf_hub_download(REPO, "0_WordEmbeddings/whitespacetokenizer_config.json")))["vocab"]
    weights = torch.load(hf_hub_download(REPO, "0_WordEmbeddings/pytorch_model.bin"),
                         map_location="cpu", weights_only=True)["emb_layer.weight"]
    return {word: i for i, word in enumerate(vocab)}, weights.numpy().astype(np.float32)


def average_embeddings(token_lists, word_to_row, matrix):
    """
    One vector per text: the mean of the embeddings of its tokens. Tokens without an embedding are skipped;
    a text with no known token gets the zero vector.

    token_lists: one list of (lowercased) tokens per text
    """
    out = np.zeros((len(token_lists), matrix.shape[1]), dtype=np.float32)
    for r, tokens in enumerate(token_lists):
        rows = [word_to_row[t] for t in tokens if t in word_to_row]
        if rows:
            out[r] = matrix[rows].mean(axis=0)
    return out


def coverage(token_lists, word_to_row):
    """Share of tokens that have an embedding."""
    n_tokens = sum(len(t) for t in token_lists)
    return sum(t in word_to_row for tokens in token_lists for t in tokens) / n_tokens
