"""
01_defamiliarization_analysis.py
==============================
Fine-tunes a GPT-2 language model on the ELTeC corpus of historical novels to
establish a stylometric baseline of "familiar" 19th-century prose.  Then
measures how much each test novel (stored in test_novels/) departs from that
baseline via perplexity — operationalising Shklovsky's concept of
defamiliarization (ostranenie) in a computational framework.

Outputs (CSVs + PNGs) are written to the directory named in CONFIG['output_dir'].

Run modes
---------
  FAST   — 1 ELTeC book,  100 sentences, 1 epoch  (~3-5 min)
  FASTER — 2 ELTeC books, 500 sentences, 3 epochs (~15-25 min)
  FULL   — ALL ELTeC books, ALL sentences, 4 epochs (can take hours)

Resumption behaviour
--------------------
* Fine-tuned model: reused when the saved training metadata matches the current
  MODE.  A mismatch (or missing metadata) triggers a fresh fine-tuning run.
* Sentence-level perplexity CSVs: reused when they already exist on disk AND
  the computation phase is being skipped or the model is being reused.  Set
  FORCE_RECOMPUTE_SENTENCES = True to override.
"""

from __future__ import annotations

import json
import os
import glob
import re
import math
import time
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Execution toggles
# ---------------------------------------------------------------------------
RUN_COMPUTATION           = True   # False → skip training; re-use existing CSVs
RUN_VISUALIZATION         = True   # False → skip chart generation
FORCE_RECOMPUTE_SENTENCES = False  # True → recompute sentence PPL even if CSVs exist

# ---------------------------------------------------------------------------
# Mode selection
# ---------------------------------------------------------------------------
# Set MODE to one of: 'FAST', 'FASTER', or 'FULL'
MODE = 'FASTER'

_FAST_CONFIG = {
    'eltec_sample_count':  1,       # number of ELTeC books to sample (None = ALL)
    'epochs':              1,
    'max_length':          128,     # token context window
    'sentence_sample_size': 100,    # sentences per corpus (None = ALL)
    'train_char_limit':    100_000, # character cap on training text (None = unlimited)
    'eval_char_limit':     50_000,  # character cap on eval text    (None = unlimited)
}

_FASTER_CONFIG = {
    'eltec_sample_count':  2,
    'epochs':              3,
    'max_length':          256,
    'sentence_sample_size': 500,
    'train_char_limit':    300_000,
    'eval_char_limit':     150_000,
}

_FULL_CONFIG = {
    'eltec_sample_count':  None,    # None = use ALL available ELTeC books
    'epochs':              4,
    'max_length':          256,
    'sentence_sample_size': None,   # None = use ALL sentences
    'train_char_limit':    None,    # None = no character limit
    'eval_char_limit':     None,
}

_PRESETS = {'FAST': _FAST_CONFIG, 'FASTER': _FASTER_CONFIG, 'FULL': _FULL_CONFIG}
if MODE not in _PRESETS:
    raise ValueError(f"MODE must be one of {list(_PRESETS.keys())}, got '{MODE}'")
_mode_params = _PRESETS[MODE]

CONFIG = {
    'eltec_dir':                  'ELTEC',
    'test_dir':                   'test_novels',
    'output_dir':                 'results',
    'model_dir':                  'models',
    'model_name':                 'gpt2',       # base GPT-2 (117 M parameters)
    'finetuned_model_subdir':     'gpt2_eltec_finetuned',
    'batch_size':                 4,
    'gradient_accumulation_steps': 4,
    'learning_rate':              5e-5,
    'device': torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    ),
    **_mode_params,
}

# Derived convenience paths (set once; never recomputed inline)
_MODEL_SAVE_PATH = os.path.join(CONFIG['model_dir'], CONFIG['finetuned_model_subdir'])
_METADATA_PATH   = os.path.join(_MODEL_SAVE_PATH, 'training_metadata.json')

# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalise whitespace and strip non-ASCII characters."""
    text = str(text)
    text = re.sub(r'\s+', ' ', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text.strip()


import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    
def split_into_sentences(text: str) -> list[str]:
    """Split *text* on sentence boundaries naturally using NLTK."""
    sentences = nltk.tokenize.sent_tokenize(text)
    return [s.strip() for s in sentences if len(s.strip()) > 5]


def get_stylistic_features(text: str) -> tuple[int, float, float]:
    """Return (word_count, avg_word_length, type_token_ratio) for *text*."""
    words = text.split()
    word_count = len(words)
    if word_count == 0:
        return 0, 0.0, 0.0
    avg_word_length = sum(len(w) for w in words) / word_count
    type_token_ratio = len(set(words)) / word_count
    return word_count, avg_word_length, type_token_ratio

# ---------------------------------------------------------------------------
# Corpus loaders
# ---------------------------------------------------------------------------

def load_eltec_corpus(eltec_dir: str, sample_count: int | None) -> tuple[str, int, list[str]]:
    """Load and concatenate ELTeC training texts.

    Parameters
    ----------
    eltec_dir:    Path to folder containing ELTeC .txt files.
    sample_count: How many files to sample.  None means all.

    Returns
    -------
    (concatenated_text, actual_book_count, list_of_filenames)
    """
    print("Loading ELTeC training corpus...")
    all_eltec_files = glob.glob(os.path.join(eltec_dir, '*.txt'))

    if not all_eltec_files:
        print(f"Warning: No .txt files found in '{eltec_dir}'.")
        return "", 0, []

    if sample_count is None:
        selected_files = all_eltec_files
    else:
        sample_count   = min(sample_count, len(all_eltec_files))
        selected_files = random.sample(all_eltec_files, sample_count)

    selected_names = [os.path.basename(f) for f in selected_files]
    actual_count   = len(selected_files)

    print(f"Using {actual_count} out of {len(all_eltec_files)} ELTeC books:")
    for name in selected_names:
        print(f"  - {name}")

    eltec_texts = []
    for file_path in tqdm(selected_files, desc="Reading selected ELTeC files"):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as fh:
                eltec_texts.append(clean_text(fh.read()))
        except Exception as exc:
            print(f"Error reading '{file_path}': {exc}")

    return " ".join(eltec_texts), actual_count, selected_names


def load_test_corpora(test_dir: str) -> dict[str, str]:
    """Load all .txt files from *test_dir* into a {novel_name: text} dict."""
    print(f"Loading test novels from '{test_dir}'...")
    test_files = glob.glob(os.path.join(test_dir, '*.txt'))

    if not test_files:
        print(f"Warning: No .txt files found in '{test_dir}'.")
        return {}

    corpora = {}
    for file_path in test_files:
        novel_name = os.path.splitext(os.path.basename(file_path))[0]
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as fh:
                corpora[novel_name] = clean_text(fh.read())
            print(f"  - Loaded {novel_name}")
        except Exception as exc:
            print(f"Error reading '{file_path}': {exc}")

    return corpora

# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class TokenBlockDataset(Dataset):
    """Tokenises *text* and splits it into fixed-length blocks for LM training."""

    def __init__(self, text: str, tokenizer, max_length: int):
        self.input_ids: list[torch.Tensor] = []

        chunk_size = 1_000_000
        for chunk_start in range(0, len(text), chunk_size):
            chunk  = text[chunk_start : chunk_start + chunk_size]
            tokens = tokenizer.encode(chunk, add_special_tokens=False)
            for block_start in range(0, len(tokens) - max_length + 1, max_length):
                self.input_ids.append(
                    torch.tensor(tokens[block_start : block_start + max_length])
                )

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.input_ids[idx]

# ---------------------------------------------------------------------------
# Perplexity computation
# ---------------------------------------------------------------------------

def compute_corpus_perplexity(model, dataloader, device) -> float:
    """Compute average cross-entropy loss over *dataloader* and return PPL."""
    model.eval()
    total_loss  = 0.0
    total_steps = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating perplexity", leave=False):
            inputs  = batch.to(device)
            outputs = model(inputs, labels=inputs)
            total_loss  += outputs.loss.item()
            total_steps += 1

    if total_steps == 0:
        return float('inf')
    avg_loss = total_loss / total_steps
    try:
        return math.exp(avg_loss)
    except OverflowError:
        return float('inf')


def compute_sentence_perplexities(model, tokenizer, sentences: list[str], device) -> list[float]:
    """Return a per-sentence perplexity list aligned with *sentences*."""
    model.eval()
    perplexities = []
    with torch.no_grad():
        for sentence in tqdm(sentences, desc="Computing sentence perplexity"):
            if not sentence.strip():
                perplexities.append(float('inf'))
                continue
            inputs = tokenizer(
                sentence, return_tensors='pt',
                truncation=True, max_length=CONFIG['max_length']
            ).to(device)
            if inputs['input_ids'].size(1) < 2:
                perplexities.append(float('nan'))
                continue
            outputs = model(**inputs, labels=inputs['input_ids'])
            try:
                ppl = math.exp(outputs.loss.item())
            except OverflowError:
                ppl = float('inf')
            perplexities.append(ppl)
    return perplexities


def compute_token_surprisal(model, tokenizer, sentence: str, device) -> list[tuple[str, float]]:
    """Return a list of (token, surprisal_bits) pairs for *sentence*."""
    model.eval()
    inputs = tokenizer(sentence, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
    logits       = outputs.logits
    input_ids    = inputs['input_ids']
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    loss_fn      = torch.nn.CrossEntropyLoss(reduction='none')
    token_losses = loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    tokens      = tokenizer.convert_ids_to_tokens(shift_labels[0])
    surprisals  = token_losses.tolist()
    return list(zip(tokens, surprisals))

# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

def fine_tune_on_eltec(
    model,
    tokenizer,
    eltec_corpus:      str,
    test_corpora_dict: dict[str, str],
    epochs:            int,
    batch_size:        int,
    accum_steps:       int,
    learning_rate:     float,
    device,
) -> tuple:
    """Fine-tune *model* on *eltec_corpus*; evaluate defamiliarization gap each epoch.

    Returns
    -------
    (model, epoch_data_dict, pre_training_stats_dict, post_training_stats_dict)
    """
    print("\nPreparing datasets...")

    # --- Training dataset ---
    train_limit  = CONFIG['train_char_limit']
    train_text   = (eltec_corpus[:train_limit]
                    if train_limit is not None and len(eltec_corpus) > train_limit
                    else eltec_corpus)
    train_dataset    = TokenBlockDataset(train_text, tokenizer, CONFIG['max_length'])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # --- Evaluation datasets ---
    eval_limit       = CONFIG['eval_char_limit']
    eval_eltec_text  = (eltec_corpus[:eval_limit]
                        if eval_limit is not None and len(eltec_corpus) > eval_limit
                        else eltec_corpus)
    eval_eltec_ds    = TokenBlockDataset(eval_eltec_text, tokenizer, CONFIG['max_length'])
    eval_eltec_dl    = DataLoader(eval_eltec_ds, batch_size=batch_size)

    test_eval_dataloaders = {}
    for novel_name, text in test_corpora_dict.items():
        eval_text = (text[:eval_limit]
                     if eval_limit is not None and len(text) > eval_limit
                     else text)
        ds = TokenBlockDataset(eval_text, tokenizer, CONFIG['max_length'])
        test_eval_dataloaders[novel_name] = DataLoader(ds, batch_size=batch_size)

    # --- Optimiser & scheduler ---
    optimizer     = AdamW(model.parameters(), lr=learning_rate)
    total_steps   = (len(train_dataloader) // accum_steps) * epochs
    scheduler     = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    epoch_data_dict = {name: [] for name in test_corpora_dict}

    # --- Pre-training evaluation ---
    print("\nEvaluating pre-training perplexity...")
    pre_eltec_ppl   = compute_corpus_perplexity(model, eval_eltec_dl, device)
    pre_stats_dict  = {}
    for novel_name, dl in test_eval_dataloaders.items():
        novel_ppl = compute_corpus_perplexity(model, dl, device)
        gap       = novel_ppl - pre_eltec_ppl
        record    = {
            'Epoch':            0,
            'ELTeC_Perplexity': pre_eltec_ppl,
            f'{novel_name}_Perplexity': novel_ppl,
            'Gap':              gap,
        }
        epoch_data_dict[novel_name].append(record)
        pre_stats_dict[novel_name] = record
        print(f"Pre-training [{novel_name}] — ELTeC PPL: {pre_eltec_ppl:.2f}, "
              f"Novel PPL: {novel_ppl:.2f}, Gap: {gap:.2f}")

    # --- Training loop ---
    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        model.train()
        model.zero_grad()
        progress_bar = tqdm(train_dataloader, desc=f"Training epoch {epoch}")

        for step, batch in enumerate(progress_bar):
            inputs  = batch.to(device)
            outputs = model(inputs, labels=inputs)
            loss    = outputs.loss / accum_steps
            loss.backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            progress_bar.set_postfix({'loss': f'{loss.item() * accum_steps:.4f}'})

        print("Evaluating after epoch...")
        epoch_eltec_ppl = compute_corpus_perplexity(model, eval_eltec_dl, device)
        for novel_name, dl in test_eval_dataloaders.items():
            novel_ppl = compute_corpus_perplexity(model, dl, device)
            gap       = novel_ppl - epoch_eltec_ppl
            record    = {
                'Epoch':            epoch,
                'ELTeC_Perplexity': epoch_eltec_ppl,
                f'{novel_name}_Perplexity': novel_ppl,
                'Gap':              gap,
            }
            epoch_data_dict[novel_name].append(record)
            print(f"Epoch {epoch} [{novel_name}] — ELTeC PPL: {epoch_eltec_ppl:.2f}, "
                  f"Novel PPL: {novel_ppl:.2f}, Gap: {gap:.2f}")

    post_stats_dict = {name: epoch_data_dict[name][-1] for name in test_corpora_dict}
    return model, epoch_data_dict, pre_stats_dict, post_stats_dict

# ---------------------------------------------------------------------------
# Output: CSV tables
# ---------------------------------------------------------------------------

def save_analysis_tables(
    novel_name:       str,
    epoch_df:         pd.DataFrame,
    novel_doc_ppl:    float,
    baseline_doc_ppl: float,
    top_sentences_df: pd.DataFrame,
    sentence_stats_df: pd.DataFrame,
    eltec_df:         pd.DataFrame,
    novel_sentences_df: pd.DataFrame,
    output_dir:       str,
    pre_training_stats: dict,
    post_training_stats: dict,
) -> None:
    """Write all analysis CSVs for one test novel to *output_dir*."""

    # Table 1 — Document-level comparison
    table1 = pd.DataFrame({
        'Corpus': ['ELTeC (Baseline)', novel_name],
        'Document_Level_Perplexity': [baseline_doc_ppl, novel_doc_ppl],
    })
    table1['Absolute_Difference_from_Baseline'] = (
        table1['Document_Level_Perplexity'] - baseline_doc_ppl
    )
    table1.to_csv(
        os.path.join(output_dir, f'{novel_name}_Table1_Document_Level_Comparison.csv'),
        index=False,
    )

    # Table 2 — Epoch-by-epoch perplexity tracking
    epoch_df.to_csv(
        os.path.join(output_dir, f'{novel_name}_Table2_Epoch_Tracking.csv'),
        index=False,
    )

    # Table 3 — Top high-perplexity (defamiliarization) moments
    top_sentences_df.to_csv(
        os.path.join(output_dir, f'{novel_name}_Table3_Top_Defamiliarization_Moments.csv'),
        index=False,
    )

    # Table 4 — Statistical summary (ELTeC + novel combined)
    eltec_stats = {
        'Corpus':          'ELTeC',
        'Mean_Perplexity': eltec_df['Perplexity'].mean(),
        'Median':          eltec_df['Perplexity'].median(),
        'Std_Dev':         eltec_df['Perplexity'].std(),
        'Pct_95':          eltec_df['Perplexity'].quantile(0.95),
        'Max':             eltec_df['Perplexity'].max(),
    }
    combined_stats_df = pd.concat(
        [pd.DataFrame([eltec_stats]), sentence_stats_df], ignore_index=True
    )
    combined_stats_df.to_csv(
        os.path.join(output_dir, f'{novel_name}_Table4_Statistical_Summary.csv'),
        index=False,
    )

    # Table 5 — Chapter/analysis summary with Shklovsky framing
    gap_change = post_training_stats['Gap'] - pre_training_stats['Gap']
    table5 = pd.DataFrame({
        'Analysis_Level': ['Document', 'Sentence', 'Token'],
        'Finding': [
            f"Perplexity gap changed by {gap_change:.2f} across fine-tuning",
            f"{novel_name} sentence PPL compared to ELTeC baseline",
            "High token surprisal marks specific loci of deviation",
        ],
        'Interpretation': [
            "Macro-level divergence from 19th-century prose norm",
            "Pervasive micro-level stylistic innovation",
            "Specific linguistic sites of ostranenie",
        ],
        'Shklovsky_Connection': [
            "Prolonged perception via structural deviation",
            "Making the familiar strange sentence-by-sentence",
            "Roughened form at the level of the signifier",
        ],
    })
    table5.to_csv(
        os.path.join(output_dir, f'{novel_name}_Table5_Defamiliarization_Summary.csv'),
        index=False,
    )

    # Raw sentence data (used by visualisation engine)
    novel_sentences_df.to_csv(
        os.path.join(output_dir, f'{novel_name}_Sentence_Data.csv'),
        index=False,
    )

# ---------------------------------------------------------------------------
# Visualisation engine
# ---------------------------------------------------------------------------

def generate_visualizations(output_dir: str) -> None:
    """Read all saved CSVs in *output_dir* and render publication-quality figures."""
    print("\n--- Generating Visualizations from CSV Data ---")
    sns.set_theme(style="whitegrid")

    eltec_csv_path = os.path.join(output_dir, 'ELTeC_Baseline_Sentence_Data.csv')
    if not os.path.exists(eltec_csv_path):
        print(f"Baseline CSV not found at '{eltec_csv_path}'. Cannot generate visualizations.")
        return

    eltec_df      = pd.read_csv(eltec_csv_path)
    baseline_mean = eltec_df['Perplexity'].mean()
    all_dfs       = [eltec_df]

    sentence_csv_files = [
        f for f in glob.glob(os.path.join(output_dir, '*_Sentence_Data.csv'))
        if 'ELTeC' not in f
    ]

    for file_path in sentence_csv_files:
        novel_name   = os.path.basename(file_path).replace('_Sentence_Data.csv', '')
        print(f"  -> Rendering charts for {novel_name}...")
        sentences_df = pd.read_csv(file_path)
        all_dfs.append(sentences_df)

        # Figure 1 — Perplexity divergence during fine-tuning
        epoch_csv = os.path.join(output_dir, f'{novel_name}_Table2_Epoch_Tracking.csv')
        if os.path.exists(epoch_csv):
            epoch_df = pd.read_csv(epoch_csv)
            plt.figure(figsize=(10, 6))
            plt.plot(epoch_df['Epoch'], epoch_df['ELTeC_Perplexity'],
                     marker='o', color='steelblue', label='ELTeC (familiar baseline)')
            plt.plot(epoch_df['Epoch'], epoch_df[f'{novel_name}_Perplexity'],
                     marker='o', color='crimson', label=f'{novel_name} (defamiliarized)')
            plt.title(
                f'Defamiliarization: Perplexity Divergence During Fine-Tuning\n({novel_name})',
                fontsize=14,
            )
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Perplexity', fontsize=12)
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f'{novel_name}_Figure1_Perplexity_Divergence.png'),
                dpi=300,
            )
            plt.close()

        # Figure 2 — Perplexity landscape across sentence sequence
        plt.figure(figsize=(14, 6))
        x = sentences_df['Sentence_Number']
        y = sentences_df['Perplexity']
        plt.plot(x, y, color='mediumpurple', alpha=0.85)
        plt.fill_between(x, y, alpha=0.25, color='mediumpurple')
        plt.axhline(y=baseline_mean, color='steelblue', linestyle='--',
                    label='ELTeC baseline mean')
        for _, row in sentences_df.nlargest(4, 'Perplexity').iterrows():
            plt.annotate(
                f"#{int(row['Sentence_Number'])}",
                xy=(row['Sentence_Number'], row['Perplexity']),
                xytext=(0, 10), textcoords='offset points',
                ha='center', fontsize=9, fontweight='bold',
            )
        plt.title(
            f'Defamiliarization Landscape: Perplexity Across {novel_name}',
            fontsize=14,
        )
        plt.xlabel(f'Sentence Number (first {len(x)})', fontsize=12)
        plt.ylabel('Sentence Perplexity', fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f'{novel_name}_Figure2_Perplexity_Landscape.png'),
            dpi=300,
        )
        plt.close()

        # Figure 3 — Perplexity distribution comparison (ELTeC vs novel)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        sns.histplot(eltec_df['Perplexity'],      color='steelblue', alpha=0.5,
                     label='ELTeC',      ax=axes[0], stat='density', kde=True, log_scale=True)
        sns.histplot(sentences_df['Perplexity'],  color='crimson',   alpha=0.5,
                     label=novel_name,   ax=axes[0], stat='density', kde=True, log_scale=True)
        axes[0].set_title('Perplexity Distribution (log scale)')
        axes[0].legend()
        combined_df = pd.concat([eltec_df, sentences_df])
        sns.boxplot(data=combined_df, x='Corpus', y='Perplexity', ax=axes[1],
                    hue='Corpus', legend=False, palette=['steelblue', 'crimson'])
        axes[1].set_yscale('log')
        axes[1].set_title('Perplexity Boxplots (log scale)')
        plt.suptitle(
            f'Perplexity Distribution: ELTeC vs {novel_name}', fontsize=16
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f'{novel_name}_Figure3_Distribution_Comparison.png'),
            dpi=300,
        )
        plt.close()

        # Figure 4 — Top defamiliarization moments (horizontal bar)
        top_moments_csv = os.path.join(
            output_dir, f'{novel_name}_Table3_Top_Defamiliarization_Moments.csv'
        )
        if os.path.exists(top_moments_csv):
            top_df = pd.read_csv(top_moments_csv).sort_values('Perplexity_Score', ascending=True)
            labels = [
                f"#{int(row['Sentence_Number'])}: {str(row['Text_Excerpt'])[:40]}..."
                for _, row in top_df.iterrows()
            ]
            plt.figure(figsize=(12, 8))
            bars = plt.barh(
                labels, top_df['Perplexity_Score'],
                color=sns.color_palette("Reds", len(labels))
            )
            plt.title('Highest-Surprisal Moments (Ostranenie)', fontsize=14)
            plt.xlabel('Perplexity Score', fontsize=12)
            for bar in bars:
                w = bar.get_width()
                plt.text(w + w * 0.01, bar.get_y() + bar.get_height() / 2,
                         f'{w:.0f}', va='center')
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f'{novel_name}_Figure4_Top_Defamiliarization_Moments.png'),
                dpi=300,
            )
            plt.close()

        # Figure 5 — Rolling-average perplexity
        plt.figure(figsize=(14, 6))
        rolling_ppl = sentences_df['Perplexity'].rolling(window=50, min_periods=1).mean()
        plt.plot(sentences_df['Sentence_Number'], rolling_ppl,
                 color='darkorange', linewidth=2, label='Rolling mean (w=50)')
        plt.axhline(y=baseline_mean, color='steelblue', linestyle='--',
                    label='ELTeC baseline mean')
        plt.title(f'Rolling Perplexity (window = 50): {novel_name}', fontsize=14)
        plt.xlabel('Sentence Number', fontsize=12)
        plt.ylabel('Rolling Average Perplexity', fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f'{novel_name}_Figure5_Rolling_Perplexity.png'),
            dpi=300,
        )
        plt.close()

        # Figure 6 — Stylistic features vs perplexity
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        sns.scatterplot(data=sentences_df, x='Word_Count',     y='Perplexity',
                        ax=axes[0], color='seagreen',  alpha=0.6)
        axes[0].set_title('Perplexity vs Sentence Length')
        axes[0].set_yscale('log')
        sns.scatterplot(data=sentences_df, x='Avg_Word_Length', y='Perplexity',
                        ax=axes[1], color='teal',      alpha=0.6)
        axes[1].set_title('Perplexity vs Avg Word Length')
        axes[1].set_yscale('log')
        sns.scatterplot(data=sentences_df, x='TTR',             y='Perplexity',
                        ax=axes[2], color='slateblue', alpha=0.6)
        axes[2].set_title('Perplexity vs Type-Token Ratio')
        axes[2].set_yscale('log')
        plt.suptitle(f'Stylistic Features vs Perplexity ({novel_name})', fontsize=16)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f'{novel_name}_Figure6_Stylistics_vs_Perplexity.png'),
            dpi=300,
        )
        plt.close()

    # Global cross-novel comparison
    if all_dfs:
        global_df = pd.concat(all_dfs, ignore_index=True)
        plt.figure(figsize=(14, 8))
        sns.boxplot(data=global_df, x='Corpus', y='Perplexity',
                    hue='Corpus', legend=False, palette='Set2')
        plt.yscale('log')
        plt.title('Global Defamiliarization Comparison: All Corpora', fontsize=16)
        plt.ylabel('Sentence Perplexity (log scale)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, 'Global_Figure_Defamiliarization_Comparison.png'),
            dpi=300,
        )
        plt.close()

# ---------------------------------------------------------------------------
# Model loading / saving helpers
# ---------------------------------------------------------------------------

def _save_model_and_metadata(model, tokenizer) -> None:
    """Persist fine-tuned model weights and training metadata to disk."""
    model.save_pretrained(_MODEL_SAVE_PATH)
    tokenizer.save_pretrained(_MODEL_SAVE_PATH)
    metadata = {
        'MODE':                MODE,
        'epochs':              CONFIG['epochs'],
        'eltec_sample_count':  CONFIG['eltec_sample_count'],
        'max_length':          CONFIG['max_length'],
        'train_char_limit':    CONFIG['train_char_limit'],
        'sentence_sample_size': CONFIG['sentence_sample_size'],
    }
    with open(_METADATA_PATH, 'w') as fh:
        json.dump(metadata, fh, indent=4)
    print(f"\nFine-tuned model and metadata saved to '{_MODEL_SAVE_PATH}'")


def _load_existing_model(tokenizer_out):
    """Load an already fine-tuned model from disk.  Returns (tokenizer, model)."""
    print(f"Loading existing fine-tuned ELTeC model from '{_MODEL_SAVE_PATH}'...")
    tokenizer = GPT2Tokenizer.from_pretrained(_MODEL_SAVE_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(_MODEL_SAVE_PATH).to(CONFIG['device'])
    return tokenizer, model


def _check_model_reuse() -> bool:
    """Return True if a saved model matching the current MODE exists."""
    config_json = os.path.join(_MODEL_SAVE_PATH, 'config.json')
    if not (os.path.exists(_MODEL_SAVE_PATH) and os.path.exists(config_json)):
        return False
    if not os.path.exists(_METADATA_PATH):
        print("Saved model exists but has no metadata file. Retraining...")
        return False
    try:
        with open(_METADATA_PATH, 'r') as fh:
            saved_meta = json.load(fh)
        if saved_meta.get('MODE') == MODE:
            print(f"Found existing fine-tuned model for MODE='{MODE}' at '{_MODEL_SAVE_PATH}'.")
            return True
        print(f"Saved model was trained with MODE='{saved_meta.get('MODE')}', "
              f"current MODE='{MODE}'. Retraining...")
        return False
    except Exception as exc:
        print(f"Error reading metadata ({exc}). Retraining...")
        return False


def _load_epoch_data_from_csv(novel_name: str) -> tuple[list, dict, dict]:
    """Reconstruct epoch_data_dict, pre_stats, post_stats from a saved CSV."""
    epoch_csv = os.path.join(CONFIG['output_dir'], f'{novel_name}_Table2_Epoch_Tracking.csv')
    placeholder = {
        'Epoch': 0, 'ELTeC_Perplexity': 1.0,
        f'{novel_name}_Perplexity': 1.0, 'Gap': 0.0,
    }
    if os.path.exists(epoch_csv):
        try:
            records = pd.read_csv(epoch_csv).to_dict(orient='records')
            pre  = records[0]  if records else placeholder
            post = records[-1] if records else placeholder
            return records, pre, post
        except Exception as exc:
            print(f"Error loading epoch CSV for '{novel_name}': {exc}")
    return [placeholder], placeholder, placeholder

# ---------------------------------------------------------------------------
# Sentence-level CSV caching helpers
# ---------------------------------------------------------------------------

def _sentence_csv_exists(novel_name: str) -> bool:
    path = os.path.join(CONFIG['output_dir'], f'{novel_name}_Sentence_Data.csv')
    return os.path.exists(path)


def _eltec_sentence_csv_exists() -> bool:
    path = os.path.join(CONFIG['output_dir'], 'ELTeC_Baseline_Sentence_Data.csv')
    return os.path.exists(path)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    start_time = time.time()
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    os.makedirs(CONFIG['model_dir'],  exist_ok=True)

    mode_descriptions = {
        'FAST':   'Fast   — 1 ELTeC book,  100 sentences, 1 epoch',
        'FASTER': 'Faster — 2 ELTeC books, 500 sentences, 3 epochs',
        'FULL':   'Full   — ALL ELTeC books, ALL sentences, 4 epochs',
    }
    books_label = (CONFIG['eltec_sample_count']
                   if CONFIG['eltec_sample_count'] is not None else 'ALL')
    sents_label = (CONFIG['sentence_sample_size']
                   if CONFIG['sentence_sample_size'] is not None else 'ALL')
    train_label = (f"{CONFIG['train_char_limit']:,}"
                   if CONFIG['train_char_limit'] is not None else 'unlimited')
    eval_label  = (f"{CONFIG['eval_char_limit']:,}"
                   if CONFIG['eval_char_limit']  is not None else 'unlimited')

    print("=== Defamiliarization Analysis (GPT-2 / ELTeC) ===")
    print(f"Mode         : {MODE} — {mode_descriptions[MODE]}")
    print(f"Device       : {CONFIG['device']}")
    print(f"ELTeC books  : {books_label}")
    print(f"Epochs       : {CONFIG['epochs']}")
    print(f"Max length   : {CONFIG['max_length']} tokens")
    print(f"Sentences    : {sents_label} per corpus")
    print(f"Train limit  : {train_label} chars")
    print(f"Eval limit   : {eval_label} chars")
    print()

    if RUN_COMPUTATION:
        # --- Load corpora ---
        eltec_corpus, _eltec_count, _selected_books = load_eltec_corpus(
            CONFIG['eltec_dir'], CONFIG['eltec_sample_count']
        )
        if len(eltec_corpus.split()) < 100:
            print("Using dummy ELTeC data (no files found or corpus too small).")
            eltec_corpus = (
                "The carriage drove up to the house. The gentleman stepped out. "
                "It was a fine day in London. "
            ) * 1000

        test_corpora_dict = load_test_corpora(CONFIG['test_dir'])
        if not test_corpora_dict:
            print("No test novels found. Exiting.")
            return
        for novel_name, text in list(test_corpora_dict.items()):
            if len(text.split()) < 100:
                print(f"Using dummy data for '{novel_name}' (file too small).")
                test_corpora_dict[novel_name] = (
                    "Stately, plump Buck Mulligan came from the stairhead, bearing a bowl of "
                    "lather on which a mirror and a razor lay crossed. "
                ) * 1000

        # --- Model: reuse or fine-tune ---
        print("\nLoading GPT-2 model and tokenizer...")
        reuse_model = _check_model_reuse()

        if reuse_model:
            tokenizer, model = _load_existing_model(None)
            epoch_data_dict, pre_stats_dict, post_stats_dict = {}, {}, {}
            for novel_name in test_corpora_dict:
                records, pre, post = _load_epoch_data_from_csv(novel_name)
                epoch_data_dict[novel_name] = records
                pre_stats_dict[novel_name]  = pre
                post_stats_dict[novel_name] = post
        else:
            print("Initializing base GPT-2 model...")
            tokenizer = GPT2Tokenizer.from_pretrained(CONFIG['model_name'])
            tokenizer.pad_token = tokenizer.eos_token
            model = GPT2LMHeadModel.from_pretrained(CONFIG['model_name']).to(CONFIG['device'])

            print("\nStarting fine-tuning (learning ONLY from ELTeC)...")
            model, epoch_data_dict, pre_stats_dict, post_stats_dict = fine_tune_on_eltec(
                model, tokenizer, eltec_corpus, test_corpora_dict,
                CONFIG['epochs'], CONFIG['batch_size'],
                CONFIG['gradient_accumulation_steps'],
                CONFIG['learning_rate'], CONFIG['device'],
            )
            _save_model_and_metadata(model, tokenizer)

        # --- Baseline (ELTeC) sentence-level metrics ---
        sentence_limit = CONFIG['sentence_sample_size']
        eval_limit     = CONFIG['eval_char_limit']

        # Compute or load ELTeC baseline sentence data
        eltec_csv_path = os.path.join(CONFIG['output_dir'], 'ELTeC_Baseline_Sentence_Data.csv')
        if _eltec_sentence_csv_exists() and not FORCE_RECOMPUTE_SENTENCES:
            print("\nLoading cached ELTeC baseline sentence data...")
            eltec_df = pd.read_csv(eltec_csv_path)
            # Still need baseline_doc_ppl for Table 1
            eltec_eval_text = (
                eltec_corpus[:eval_limit]
                if eval_limit is not None and len(eltec_corpus) > eval_limit
                else eltec_corpus
            )
            eltec_ds  = TokenBlockDataset(eltec_eval_text, tokenizer, CONFIG['max_length'])
            eltec_dl  = DataLoader(eltec_ds, batch_size=4)
            print("Computing ELTeC document-level perplexity...")
            baseline_doc_ppl = compute_corpus_perplexity(model, eltec_dl, CONFIG['device'])
        else:
            print("\nComputing ELTeC baseline metrics...")
            eltec_eval_text = (
                eltec_corpus[:eval_limit]
                if eval_limit is not None and len(eltec_corpus) > eval_limit
                else eltec_corpus
            )
            eltec_ds  = TokenBlockDataset(eltec_eval_text, tokenizer, CONFIG['max_length'])
            eltec_dl  = DataLoader(eltec_ds, batch_size=4)
            baseline_doc_ppl = compute_corpus_perplexity(model, eltec_dl, CONFIG['device'])

            eltec_sentences = split_into_sentences(eltec_corpus)
            if sentence_limit is not None:
                eltec_sentences = eltec_sentences[:sentence_limit]
            eltec_ppl_scores  = compute_sentence_perplexities(
                model, tokenizer, eltec_sentences, CONFIG['device']
            )
            eltec_stylistics  = [get_stylistic_features(s) for s in eltec_sentences]
            eltec_df = pd.DataFrame({
                'Corpus':         'ELTeC',
                'Sentence_Number': range(1, len(eltec_sentences) + 1),
                'Text':           eltec_sentences,
                'Perplexity':     eltec_ppl_scores,
                'Word_Count':     [s[0] for s in eltec_stylistics],
                'Avg_Word_Length': [s[1] for s in eltec_stylistics],
                'TTR':            [s[2] for s in eltec_stylistics],
            })
            eltec_df = eltec_df.replace([np.inf, -np.inf], np.nan).dropna()
            eltec_df.to_csv(eltec_csv_path, index=False)

        # --- Per-novel analysis ---
        for novel_name, novel_text in test_corpora_dict.items():
            print(f"\n--- Analyzing test novel: {novel_name} ---")

            # Document-level perplexity
            novel_eval_text = (
                novel_text[:eval_limit]
                if eval_limit is not None and len(novel_text) > eval_limit
                else novel_text
            )
            novel_ds  = TokenBlockDataset(novel_eval_text, tokenizer, CONFIG['max_length'])
            novel_dl  = DataLoader(novel_ds, batch_size=4)
            novel_doc_ppl = compute_corpus_perplexity(model, novel_dl, CONFIG['device'])

            # Sentence-level: load from cache or recompute
            novel_sentence_csv = os.path.join(
                CONFIG['output_dir'], f'{novel_name}_Sentence_Data.csv'
            )
            if _sentence_csv_exists(novel_name) and not FORCE_RECOMPUTE_SENTENCES:
                print(f"  Loading cached sentence data for '{novel_name}'...")
                novel_sentences_df = pd.read_csv(novel_sentence_csv)
                sentences = novel_sentences_df['Text'].tolist()
            else:
                sentences = split_into_sentences(novel_text)
                if sentence_limit is not None:
                    sentences = sentences[:sentence_limit]
                ppl_scores = compute_sentence_perplexities(
                    model, tokenizer, sentences, CONFIG['device']
                )
                stylistics = [get_stylistic_features(s) for s in sentences]
                novel_sentences_df = pd.DataFrame({
                    'Corpus':         novel_name,
                    'Sentence_Number': range(1, len(sentences) + 1),
                    'Text':           sentences,
                    'Perplexity':     ppl_scores,
                    'Word_Count':     [s[0] for s in stylistics],
                    'Avg_Word_Length': [s[1] for s in stylistics],
                    'TTR':            [s[2] for s in stylistics],
                })
                novel_sentences_df = (
                    novel_sentences_df.replace([np.inf, -np.inf], np.nan).dropna()
                )

            # Top defamiliarization moments
            top_10 = novel_sentences_df.nlargest(10, 'Perplexity')
            top_rows = []
            for _, row in top_10.iterrows():
                idx = int(row['Sentence_Number']) - 1
                ctx_before = sentences[idx - 1] if idx > 0         else ""
                ctx_after  = sentences[idx + 1] if idx < len(sentences) - 1 else ""
                top_rows.append({
                    'Sentence_Number': row['Sentence_Number'],
                    'Perplexity_Score': row['Perplexity'],
                    'Text_Excerpt':    row['Text'][:100],
                    'Context': (
                        f"...{ctx_before[-50:]} "
                        f"[{row['Text'][:50]}...] "
                        f"{ctx_after[:50]}..."
                    ),
                })
            top_sentences_df = pd.DataFrame(top_rows)

            # Sentence-level statistics
            sentence_stats_df = pd.DataFrame([{
                'Corpus':          novel_name,
                'Mean_Perplexity': novel_sentences_df['Perplexity'].mean(),
                'Median':          novel_sentences_df['Perplexity'].median(),
                'Std_Dev':         novel_sentences_df['Perplexity'].std(),
                'Pct_95':          novel_sentences_df['Perplexity'].quantile(0.95),
                'Max':             novel_sentences_df['Perplexity'].max(),
            }])

            # Token surprisal for highest-PPL sentence
            print("\nToken surprisal for highest-perplexity sentence:")
            if not top_10.empty:
                highest_sentence  = top_10.iloc[0]['Text']
                surprisal_pairs   = compute_token_surprisal(
                    model, tokenizer, highest_sentence, CONFIG['device']
                )
                for token, surprisal in surprisal_pairs[:10]:
                    print(f"  {token}: {surprisal:.2f}")

            epoch_df = pd.DataFrame(epoch_data_dict[novel_name])
            print(f"\nSaving analysis tables for '{novel_name}'...")
            save_analysis_tables(
                novel_name, epoch_df,
                novel_doc_ppl, baseline_doc_ppl,
                top_sentences_df, sentence_stats_df,
                eltec_df, novel_sentences_df,
                CONFIG['output_dir'],
                pre_stats_dict[novel_name],
                post_stats_dict[novel_name],
            )

    if RUN_VISUALIZATION:
        generate_visualizations(CONFIG['output_dir'])

    elapsed_minutes = (time.time() - start_time) / 60
    print("\n" + "=" * 50)
    print("DEFAMILIARIZATION ANALYSIS — OUTPUT SUMMARY")
    print("=" * 50)
    print(f"Total execution time : {elapsed_minutes:.2f} minutes")
    print(f"Output directory     : '{CONFIG['output_dir']}/'")
    print(f"Model saved at       : '{_MODEL_SAVE_PATH}/'")


if __name__ == "__main__":
    main()
