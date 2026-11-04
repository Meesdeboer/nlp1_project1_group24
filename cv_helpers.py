import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import embeddings
from from_tutorial import smlp
from from_tutorial.experiment import cross_validation_split

RESULTS_DIR = Path(__file__).parent / "results"


def make_features(ids, vocab_size, indicator):
    """
    Sparse feature matrix straight from token ids (no dense vector per text).

    ids: one list/array of token ids per text
    indicator: True -> token indicators (0/1), False -> bag of words (counts)
    """
    lengths = np.array([len(x) for x in ids])
    rows = np.repeat(np.arange(len(ids)), lengths)
    cols = np.concatenate([np.asarray(x, dtype=int) for x in ids])
    # duplicate (row, col) entries are summed when converting to CSR, which gives the counts
    X = csr_matrix((np.ones(len(cols), dtype=np.float32), (rows, cols)), shape=(len(ids), vocab_size))
    if indicator:
        X.data[:] = 1
    return X


def make_folds(y, k, seed):
    """Tutorial's cross_validation_split, returned as (train_idx, val_idx) integer arrays."""
    folds = cross_validation_split(list(zip(range(len(y)), y)), k, seed=seed)
    return [
        (np.array(f["training_ids"], dtype=int), np.array(f["validation_ids"], dtype=int))
        for f in folds.values()
    ]


def cv_loglinear(X, y, folds, max_iter=100, scale=False):
    """
    Accuracy per fold of a log-linear model (same settings as the tutorial's loglin.fit).
    scale: standardise the features with the mean and spread of the fold's training rows (for dense features)
    """
    return [eval_loglinear(X[train_idx], y[train_idx], X[val_idx], y[val_idx], max_iter, scale)
            for train_idx, val_idx in tqdm(folds, desc="folds")]


def cv_mlp(X, y, folds, texts=None, hidden_dim=128, dropout=0.1, batch_size=64, learning_rate=1e-4,
           weight_decay=1e-4, num_epochs=10, seed=0, device="cpu", epochs_out=None, scale=False):
    """
    Accuracy per fold of a single-hidden-layer MLP (the tutorial's SparseMLP and training loop).

    The epoch is chosen on an inner validation slice of the fold's training rows (see inner_split),
    by accuracy, so the held-out fold is scored once and plays no part in the choice.

    texts: the raw train sentences, to build the inner slice from whole parent sentences (use with parent folds);
        None picks the slice at random (use with random folds)
    epochs_out: optional list; the chosen epoch (1-based) of every fold is appended to it
    scale: standardise dense features with the mean and spread of the fold's training rows
    """
    accuracies = []
    for fold, (train_idx, val_idx) in enumerate(tqdm(folds, desc="folds")):
        torch.manual_seed(seed * 1000 + fold)
        accuracy, epoch, _ = _train_and_select(
            X, y, train_idx, X[val_idx], y[val_idx], texts, seed * 1000 + fold,
            hidden_dim, dropout, batch_size, learning_rate, weight_decay, num_epochs, device, scale=scale)
        accuracies.append(accuracy)
        if epochs_out is not None:
            epochs_out.append(epoch)
    return accuracies


def save_results(name, accuracies, **info):
    """Write accuracies (shape [seeds, folds]) and run info to results/{name}.json."""
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / f"{name}.json", "w") as f:
        json.dump({**info, "accuracies": np.asarray(accuracies).tolist()}, f, indent=2)


def eval_loglinear(X_train, y_train, X_eval, y_eval, max_iter=100, scale=False):
    """Train a log-linear model on all of X_train and return its accuracy on X_eval."""
    if scale:
        scaler = StandardScaler().fit(X_train)
        X_train, X_eval = scaler.transform(X_train), scaler.transform(X_eval)
    model = LogisticRegression(solver="saga", max_iter=max_iter)
    model.fit(X_train, y_train)
    return accuracy_score(y_eval, model.predict(X_eval))


def eval_svm(X_train, y_train, X_eval, y_eval, kernel="rbf", C=1.0, gamma="scale", scale=True):
    """
    Train an SVM on X_train and return its accuracy on X_eval.

    kernel: "linear" (liblinear's LinearSVC, fast on sparse features) or "rbf" (exact kernel SVM, for dense features)
    gamma: width of the RBF kernel, ignored by the linear SVM
    scale: standardise dense features with the mean and spread of the training rows (not for sparse features)
    """
    if scale:
        scaler = StandardScaler().fit(X_train)
        X_train, X_eval = scaler.transform(X_train), scaler.transform(X_eval)
    if kernel == "linear":
        model = LinearSVC(C=C, max_iter=5000)
    else:
        model = SVC(kernel=kernel, C=C, gamma=gamma)
    model.fit(X_train, y_train)
    return accuracy_score(y_eval, model.predict(X_eval))


def cv_svm(X, y, folds, kernel="rbf", C=1.0, gamma="scale", scale=True):
    """Accuracy per fold of an SVM (see eval_svm)."""
    return [eval_svm(X[train_idx], y[train_idx], X[val_idx], y[val_idx], kernel, C, gamma, scale)
            for train_idx, val_idx in tqdm(folds, desc="folds")]


def eval_mlp(X_train, y_train, X_eval, y_eval, texts=None, hidden_dim=128, dropout=0.1, batch_size=64,
             learning_rate=1e-4, weight_decay=1e-4, num_epochs=10, seed=0, device="cpu", scale=False):
    """
    Train the MLP on the train rows and return its accuracy on X_eval at the epoch chosen on an inner
    validation slice of the train rows (as in cv_mlp). X_eval plays no part in the choice.
    """
    torch.manual_seed(seed)
    accuracy, _, _ = _train_and_select(
        X_train, y_train, np.arange(len(y_train)), X_eval, y_eval, texts, seed,
        hidden_dim, dropout, batch_size, learning_rate, weight_decay, num_epochs, device, scale=scale)
    return accuracy


def load_results(name):
    """Fold accuracies (shape [seeds, folds]) saved by save_results."""
    with open(RESULTS_DIR / f"{name}.json") as f:
        return np.array(json.load(f)["accuracies"])


def sub_spans(words):
    """All proper contiguous word sequences of a row (shorter than the row itself)."""
    n = len(words)
    for i in range(n):
        for j in range(i + 1, n + 1):
            if j - i < n:
                yield " ".join(words[i:j])


_PARENT_CACHE = {}


def find_parents(texts):
    """
    Indices of the rows that are not inside any longer row. In SST-2 train these are the original
    sentences; all other rows are their sub-phrases. Cached on disk, as it takes about a minute.
    """
    digest = hashlib.md5("\n".join(texts).encode()).hexdigest()
    if digest in _PARENT_CACHE:
        return _PARENT_CACHE[digest]
    cache_file = Path(__file__).parent / "data" / "train_parents.json"
    if cache_file.exists():
        cached = json.loads(cache_file.read_text())
        if cached["hash"] == digest:
            _PARENT_CACHE[digest] = np.array(cached["parents"], dtype=int)
            return _PARENT_CACHE[digest]

    inside_a_longer_row = set()
    for t in texts:
        inside_a_longer_row.update(sub_spans(t.split()))
    parents = np.array([i for i, t in enumerate(texts) if t not in inside_a_longer_row], dtype=int)

    cache_file.parent.mkdir(exist_ok=True)
    cache_file.write_text(json.dumps({"hash": digest, "parents": parents.tolist()}))
    _PARENT_CACHE[digest] = parents
    return parents


def parent_folds(texts, y, k, seed):
    """
    Cross-validation folds without phrase leakage, as (train_idx, heldout_idx) pairs over the rows of texts.

    The held-out fold is a random set of parent sentences (rows that are not inside a longer row).
    The training fold is every other row except those equal to, or inside, a held-out parent,
    so the model never sees any part of the text it is tested on.

    texts: the raw train sentences (word-tokenised, as in SST-2), one per row
    y: labels, one per row
    """
    y = np.asarray(y)
    parents = find_parents(texts)
    folds = []
    for _, held_out in make_folds(y[parents], k, seed):
        heldout_idx = parents[held_out]
        blocked = _text_and_spans(texts, heldout_idx)
        train_idx = np.array([i for i, t in enumerate(texts) if t not in blocked], dtype=int)
        folds.append((train_idx, heldout_idx))
    return folds


def mlp_curve(X, y, train_idx, val_idx, texts=None, hidden_dim=128, dropout=0.1, batch_size=64,
              learning_rate=1e-4, weight_decay=1e-4, num_epochs=10, seed=0, device="cpu", scale=False):
    """
    Train the MLP on one fold exactly as cv_mlp does (fold 0 of seed `seed`) and return, for every epoch,
    the loss/accuracy on the training rows ("train", averaged over the epoch with dropout on), on the inner
    validation slice ("select") and on the held-out fold ("heldout").
    """
    torch.manual_seed(seed * 1000)
    _, _, history = _train_and_select(
        X, y, train_idx, X[val_idx], y[val_idx], texts, seed * 1000,
        hidden_dim, dropout, batch_size, learning_rate, weight_decay, num_epochs, device, record_history=True, scale=scale)
    return history


def _text_and_spans(texts, idx):
    """The texts of rows idx together with all their sub-spans."""
    blocked = set()
    for i in idx:
        blocked.add(texts[i])
        blocked.update(sub_spans(texts[i].split()))
    return blocked


def inner_split(train_idx, texts=None, frac=0.1, seed=0):
    """
    Split the training rows of one fold into (fit_idx, select_idx): the rows to train on and a slice
    used only to choose the epoch.

    With texts (parent folds) the slice is a random 10% of the parent sentences, and every row equal to or inside
    one of them is removed from fit_idx, so the slice is as leak-free as the held-out fold.
    Without texts (random folds) the slice is a random 10% of the rows.
    """
    rng = np.random.default_rng(seed)
    train_idx = np.asarray(train_idx)
    if texts is None:
        shuffled = rng.permutation(train_idx)
        n = int(len(shuffled) * frac)
        return np.sort(shuffled[n:]), np.sort(shuffled[:n])
    parents = np.intersect1d(find_parents(texts), train_idx)
    select_idx = np.sort(rng.choice(parents, size=int(len(parents) * frac), replace=False))
    blocked = _text_and_spans(texts, select_idx)
    fit_idx = np.array([i for i in train_idx if texts[i] not in blocked], dtype=int)
    return fit_idx, select_idx


def _train_and_select(X, y, train_idx, eval_X, eval_y, texts, split_seed, hidden_dim, dropout, batch_size,
                      learning_rate, weight_decay, num_epochs, device, record_history=False, scale=False):
    """
    Train the MLP on the fit part of train_idx, choose the epoch with the best accuracy on the inner
    validation slice (the earliest one on ties), and return (accuracy on the eval set at that epoch,
    that epoch (1-based), history). The eval set is only used for reporting.
    """
    fit_idx, select_idx = inner_split(train_idx, texts, seed=split_seed)
    if scale:   # dense features: standardise, then store as a (full) sparse matrix for the tutorial's data loader
        scaler = StandardScaler().fit(X[train_idx])
        prep = lambda A: csr_matrix(scaler.transform(A))
    else:
        prep = lambda A: A
    loader = lambda X_, y_, shuffle: DataLoader(smlp.SparseDataset(prep(X_), y_), batch_size=batch_size,
                                                shuffle=shuffle, collate_fn=smlp.sparse_collate)
    fit_loader = loader(X[fit_idx], y[fit_idx], True)
    select_loader = loader(X[select_idx], y[select_idx], False)
    eval_loader = loader(eval_X, eval_y, False)

    model = smlp.SparseMLP(X.shape[1], hidden_dim, len(np.unique(y)), dropout).to(device)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_select, chosen_epoch, chosen_accuracy, history = -1.0, None, None, []
    for epoch in range(1, num_epochs + 1):
        train_metrics = smlp.train_one_epoch(model, fit_loader, optimizer, criterion, device)
        select_metrics, _ = smlp.evaluate(model, select_loader, criterion, device)
        improved = select_metrics["accuracy"] > best_select
        eval_metrics = smlp.evaluate(model, eval_loader, criterion, device)[0] if (improved or record_history) else None
        if improved:
            best_select, chosen_epoch, chosen_accuracy = select_metrics["accuracy"], epoch, eval_metrics["accuracy"]
        history.append({"train": train_metrics, "select": select_metrics, "heldout": eval_metrics})
    return chosen_accuracy, chosen_epoch, history


_GLOVE = {}


def build_features(df, tokenizer, features):
    """
    Feature matrix of one split for one representation, and whether it should be standardised.

    df: a split from tokenized_data.load (columns ids and tokens); tokenizer: its Tokenizer
    features: 'indicator' or 'bow' (sparse, from the token ids) or 'glove' (average GloVe embedding of the NLTK tokens)
    """
    if features == "glove":
        if not _GLOVE:
            _GLOVE["glove"] = embeddings.load_glove()
        word_to_row, matrix = _GLOVE["glove"]
        return embeddings.average_embeddings(df.tokens, word_to_row, matrix), True
    return make_features(df.ids, tokenizer.get_vocab_size(), features == "indicator"), False


def features_for(tokenizer_name, features):
    """The representations that exist for a tokenizer: GloVe vectors only exist for the NLTK (word-level) tokens."""
    return [f for f in features if f != "glove" or tokenizer_name == "nltk"]


def cached_results(name, require=(), **info):
    """
    The saved accuracies of results/{name}.json if they were made with the same settings, else None.
    Every key in info that was saved with the results must match; keys in `require` must have been saved.
    """
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        return None
    saved = json.loads(path.read_text())
    if any(key not in saved for key in require):
        return None
    if any(key in saved and saved[key] != json.loads(json.dumps(value)) for key, value in info.items()):
        return None
    return np.array(saved["accuracies"])
