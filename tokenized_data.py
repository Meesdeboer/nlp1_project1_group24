from pathlib import Path

import pandas as pd
from tokenizers import Tokenizer

DATA_DIR = Path(__file__).parent / "data" / "tokenized"


def load(name):
    """
    Loads the tokenized SST-2 data saved by data_analysis.ipynb.

    name: 'nltk', 'bpe_500', 'bpe_2000' or 'bpe_8000'
    Returns (splits, tokenizer): splits is a dict of DataFrames ('train', 'validation', 'test')
    with columns text, label, tokens, ids; tokenizer is a Hugging Face Tokenizer.
    Test labels are -1 (hidden).
    """
    folder = DATA_DIR / name
    splits = {s: pd.read_parquet(folder / f"{s}.parquet") for s in ["train", "validation", "test"]}
    return splits, Tokenizer.from_file(str(folder / "tokenizer.json"))
