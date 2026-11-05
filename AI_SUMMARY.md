# Project 1 – summary of the work (AI-generated)

*This summary was written by an AI assistant (Claude, Anthropic) on 5 November 2026, for Dunja. It covers the whole repository: the work the group did before, and the work done in one session with the assistant. All numbers come from the notebooks and the files in `results/`. Section 8 says which parts were done with AI help, for the AI disclosure in the blog post.*

---

## 1. Task and setup

- **Dataset:** SST-2 (Stanford Sentiment Treebank, binary). Sentences from Rotten Tomatoes movie reviews are labelled negative (0) or positive (1).
- **Splits:**
  - **train:** 67,349 rows, which are about 6,900 sentences plus all their labelled sub-phrases;
  - **validation:** 872 whole sentences;
  - **test:** 1,821 sentences with hidden labels, so unusable.
- **How the splits are used:** the validation split is our held-out test set, and every comparison between models is done with cross-validation on train.
- **Metric:** accuracy. Validation is almost balanced (50.9% positive), and the majority baseline is 0.509.
- **Statistical comparison:** the Bayesian correlated t-test from the tutorial, with a ROPE of 0.005. A comparison counts as decided when P(A better), P(tie) or P(B better) is above 0.9.

## 2. Where everything is

| File | Contents |
|---|---|
| `data_analysis.ipynb` | Data statistics, sub-phrase structure, tokeniser comparison, the tokenised data. Added: dataset summary, tokeniser choices, feature representations |
| `loglinear.ipynb`, `single_hidden_layer_mlp.ipynb`, `svm.ipynb` | One architecture each: 5 tokenisers x 2-3 representations, cross-validation (random vs parent folds), validation split |
| `cross_analysis.ipynb` (new) | Architectures compared, the ROPE, 3 seeds, MLP tuning, 2-layer MLP, SVM tuning, tokeniser leak, final comparison, error analysis, conclusions |
| `architectures_and_hyperparameters.ipynb` (new) | Activation functions and dropout rates |
| `cv_helpers.py` | Shared code: features, folds, models. Added: two-layer MLP, early stopping, activation option, GloVe bug fix |
| `results/` | All cached scores. `xa_*` = cross-analysis, `ah_*` = activations/dropout |

**Run order:** `data_analysis` → the three model notebooks → `cross_analysis` → `architectures_and_hyperparameters`. Everything is cached, so a rerun takes minutes. Recomputing everything from scratch takes several hours.

## 3. The homework questions, one by one

### Checklist of the brief

| Requirement | Done? | How |
|---|---|---|
| Compare text classification systems for our task | Yes | 18 systems on NLTK tokens, plus all tokenisers in the model notebooks |
| Representations: token indicators, bag of words, plus one of our own | Yes | Our own: averaged GloVe embeddings (300 dimensions) |
| Architectures: log-linear, one-hidden-layer MLP, plus one of our own | Yes | Linear SVM and a two-hidden-layer MLP, plus activation and dropout variants of the MLP |
| Comparisons based on cross-validation and a relevant metric | Yes | 5-fold "parent" cross-validation, 3 seeds, accuracy |
| Bayesian estimation with a motivated ROPE | Yes | Correlated t-test with ROPE 0.005, motivated, plus a sensitivity check at 0.0025 and 0.01 |

### Guide question 1: the dataset
- **Labels:** train rows are 55.8% positive, validation 50.9%.
- **Data quality:** 371 duplicate rows, 5 conflicting labels, and no overlap between validation and train.
- **The big finding:** train contains every sentence *and all its sub-phrases* (about 10 rows per sentence, 77% with the same label as their sentence). That's why train rows average 9 tokens and validation sentences 20. Restricted to the 6,885 parent sentences, train looks exactly like validation.
- **Vocabulary:** 5.7% of the validation tokens don't occur in train.
- **Where:** `data_analysis.ipynb`.

### Guide question 2: the tokeniser
- **What we compared:** NLTK word-level tokens, and BPE tokenisers we trained ourselves with the tutorial's `train_tokenizer` (`min_frequency=2`) at 500, 2,000, 8,000 and 20,000 tokens.
- **Why not the tutorial's tokeniser:** it was trained on AG News (news). On SST-2 validation it produces 26% more pieces than an SST-2 BPE of the same size, and more unknown pieces.
- **Training data:** SST-2 train only.
- **Is the dataset too small?** Partly. Up to about 8,000 tokens it is enough: parent sentences only and all rows give the same segmentation. 20,000 isn't supported: trained on the parent sentences only, BPE stops at 13,084 tokens. A larger similar corpus (IMDB reviews) would help the segmentation but not the classifier, and BPE 20,000 already equals NLTK.
- **Results in cross-validation (log-linear, indicators):** NLTK 0.809, BPE 20,000 0.808, BPE 8,000 0.800, BPE 2,000 0.774, BPE 500 0.713. Small vocabularies cut words into meaningless pieces. **We use NLTK.**
- **Leakage check:** building the tokenisers on all of train (including held-out folds) doesn't inflate the scores. A per-fold NLTK vocabulary gives *identical* scores, and per-fold BPE changes them by -0.001 / +0.003.

### Guide question 3: other representations
- **Indicators vs bag of words:** they're practically the same everywhere. Only 7.3% of the features have a count above 1, and those are almost all function words.
- **Our addition, averaged GloVe:**
  - **For it:** 99% coverage, it knows 85% of the words missing from the NLTK vocabulary, and similar words get similar vectors.
  - **Result:** it scores **about 2 points worse** for every architecture (log-linear in cross-validation: 0.787 vs 0.809).
  - **Why:** GloVe isn't a sentiment space (*good/bad* are as similar as *good/great*), averaging dilutes the few opinion words, and word order is lost.
- **Considered but not tried:** TF-IDF, n-grams and a learned embedding layer, with reasons given in the notebook.

### Guide question 4: architectures and hyperparameters
- **Log-linear vs MLP:** practically equal (within 0.003).
- **MLP tuning:** learning rate 1e-4/1e-3 and hidden size 32/128, chosen without leakage on an inner slice of the training data. **Didn't help:** the tuned MLP is about 0.5 points *lower*. The inner slice (about 550 sentences) is too small to choose between configurations that are all about equally good.
- **Two-hidden-layer MLP:** no better than one layer.
- **Activation functions** (ReLU, tanh, sigmoid, GELU, identity): no difference. Even with *no* non-linearity the MLP loses only about 0.5 points.
- **Dropout** (0 to 0.7): on bag of words accuracy rises steadily with dropout, up to 0.811 at 0.7, as good as log-linear. That's a trend, not decided (P = 0.77). No effect on GloVe.
- **Linear SVM:** with C=1 about 2 points worse, because it overfits. Tuning C helps on indicators (+1.1 points) but not on bag of words.

### Guide question 5: evaluation
- **Parent folds:** random cross-validation folds leak, because a held-out phrase's parent sentence is almost always in training. That inflates accuracy to **about 0.90**. With **parent folds**, a held-out sentence and all its pieces are removed from training, and cross-validation gives about 0.81, close to the validation split. This is the most important methodological point of the project.
- **Three seeds instead of one:** the differences stay the same, and the posterior widths become more reliable.
- **ROPE 0.005:** about 7 held-out sentences, half the fold-to-fold spread, and far below the differences that matter.
  - **Robust at every ROPE:** "SVM and GloVe are worse".
  - **Depends on the ROPE:** "log-linear = MLP" is undecided at 0.005 and practically equivalent at 0.01. The posterior is as wide as the ROPE itself.
- **Other aspects of performance** (`cross_analysis.ipynb`, section 9):
  - **Class bias:** five of the six systems lean positive (53-55% predicted positive vs 50.9%), inherited from the 56% positive train rows.
  - **Hard sentences:** negation -12 points, long sentences -9 to -15, contrast about -8 (borderline). Unknown words: no significant effect.
  - **Shared mistakes:** 71 sentences are wrong for every system, mostly mixed or ironic ones. The *representation* decides which sentences go wrong, not the architecture: sparse and GloVe systems disagree on 16-18% of sentences, the sparse systems among themselves on 6-11%.
  - **Log-linear weights:** *not* gets a fixed negative weight, so "not bad" scores strongly negative. *half-bad* became a positive word (it only occurs as "not half-bad"). *but* gets weight 0.
  - **Training time:** the MLPs take 3-7 times longer than the linear models, for no gain.
- **Cross-validation vs validation:** the validation sentences are slightly easier (0.6-1.9 points) than the held-out train sentences.

### Guide question 6: which model is best?
**Log-linear with token indicators on NLTK tokens.**
- It has the highest cross-validated accuracy (0.809, 3 seeds x 5 parent folds).
- It's practically tied with the MLPs and log-linear on bag of words.
- Among equal systems, it's the simplest, fastest and most interpretable.

Cross-validation (15 folds) and validation accuracy of the main systems:

| System (NLTK) | CV mean | Validation |
|---|---|---|
| log-linear, indicators | **0.809** | 0.814 |
| MLP (original), indicators | 0.809 | 0.821 |
| log-linear, bag of words | 0.808 | 0.811 |
| MLP (original), bag of words | 0.808 | 0.828 |
| two-layer MLP, indicators | 0.806 | 0.829 |
| tuned MLP, bag of words | 0.803 | 0.815 |
| tuned SVM, indicators | 0.801 | 0.813 |
| two-layer MLP, GloVe | 0.794 | 0.799 |
| SVM C=1, bag of words | 0.789 | 0.822 |
| log-linear, GloVe | 0.787 | 0.781 |

The MLPs score highest on validation, but that's a single split of 872 sentences (standard error about 1.3 points), so we don't choose on it.

## 4. What worked

- **Parent folds:** they removed a leak that made cross-validation about 9 points too optimistic. The CV estimate now matches the validation split.
- **Large BPE vocabularies** (8,000 and 20,000) match word-level tokens and have almost no unknown tokens.
- **Higher dropout** helps the bag-of-words MLP slightly, as a trend.
- **Tuning C** helps the SVM on indicators.
- **The Bayesian analysis with a ROPE** separated what's really different (SVM C=1, GloVe, small BPE) from what's practically equal (log-linear vs MLP).
- **The error analysis** explains *why* every system hits about the same ceiling.

## 5. What did not work

- **Small BPE vocabularies** (500, 2,000): 3-10 points worse.
- **Averaged GloVe:** about 2 points worse everywhere, though it makes different mistakes.
- **More MLP capacity or tuning** (learning rate, hidden size, a second layer, other activations): no gain over log-linear.
- **More seeds** didn't settle log-linear vs MLP. The differences are smaller than what 15 folds can resolve.
- **Bug, now fixed:** the GloVe features dropped every validation word stored as `[UNK]` (5.7% of the tokens). The fix raised GloVe's validation accuracy by about 1 point (log-linear 0.773 to 0.781) and doesn't change any conclusion. Cross-validation was not affected.

## 6. Conclusions

1. With bag-of-words-style features, **all reasonable architectures end up at about 81% accuracy**. Log-linear is as good as any MLP, and the non-linearity adds at most half a point.
2. **The representation matters more than the architecture:** sparse word features beat averaged GloVe, and the word-level/large-BPE tokenisers beat small BPE vocabularies.
3. **The evaluation design matters most of all:** without parent folds every number in this project would be about 9 points too high.
4. **The remaining errors are about word order:** negation, contrast and mixed or ironic sentences. The next step is features or models that see word order (n-grams, or the sequence models of projects 2 and 3), not a bigger classifier on a bag of words.

## 7. Still to do

- **Write the AI-use disclosure** for the blog post (see section 8).
- **Replace the README** (it's only ASCII art) with the run order, the GloVe download (480 MB) and the caching notes.
- **Translate the remaining Dutch notes** in `data_analysis.ipynb`.
- **Make the up to five backup slides** for the oral: data and metrics, features, architectures, results, extra.
- **Review and commit** the changes. Nothing is committed yet.

## 8. AI assistance (for the disclosure)

**Done with Claude (Anthropic) on 5 November 2026:**
- an audit of the repository against the assignment;
- environment setup and generation of the tokenised data;
- the added text and checks in `data_analysis.ipynb` (dataset summary, tokeniser choices, feature representations);
- all of `cross_analysis.ipynb` and `architectures_and_hyperparameters.ipynb`, written and run;
- the changes to `cv_helpers.py`: two-layer MLP, early stopping, activation option, `predictions_out`, and the GloVe bug fix;
- rerunning the three model notebooks after the fix;
- this summary.

The earlier work (data analysis, the three model notebooks and most of `cv_helpers.py`) was done by the group. Any AI use there should be added by whoever did it.
