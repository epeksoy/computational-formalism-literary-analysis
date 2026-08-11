"""Chapter 4 pipeline — Step 1 of 5: prepare the master analytic dataset.

Reads four CONLIT files, repairs a decimal-separator parsing artifact in the
metadata, normalizes supersense counts per 1,000 tokens, computes eight
foregrounding proxies from the unigram frequencies, and derives three
composite semantic indices. Writes one consolidated CSV.

Inputs:
  CONLIT_META.csv        11 stylometric-narratological features + BAYES + genre labels
  CONLIT_SUPERSENSE.csv  41 WordNet supersense counts (raw)
  CONLIT_LIWC.csv        117 LIWC-22 features + WC
  CONLIT_UNIGRAM.csv     10,000 top word counts

Outputs (in /mnt/user-data/outputs/):
  conlit_master.csv      one row per work; all features on the same scale
"""
import pandas as pd
import numpy as np
import unicodedata
import re
import json
import warnings

warnings.filterwarnings('ignore')

UPL = './'
OUT = 'results/'

# -------------------------------------------------------------------- constants
META_FEATS = [
    'total_characters', 'protagonist_concentration', 'avg_sentence_length',
    'avg_word_length', 'tuldava_score', 'event_count', 'speed_avg',
    'circuitousness', 'speed_min', 'volume', 'BAYES_ranking',
]

# five metadata columns known to carry a decimal-separator parsing artifact
# (a subset of rows inflated by exactly 1000x, verified: 99-100% of extremes
# land inside the core distribution after division by 1000)
ARTIFACT_COLS = ['volume', 'speed_avg', 'speed_min',
                 'circuitousness', 'protagonist_concentration']

# five embodied supersenses vs five abstract supersenses — for derived indices
EMBODIED = ['noun.body', 'verb.perception', 'verb.contact', 'verb.motion', 'verb.body']
ABSTRACT = ['noun.act', 'noun.group', 'noun.cognition', 'noun.relation', 'noun.state']

GENRE_LABEL = {
    'BS':  'Bestsellers',      'NYT': 'NYT Bestsellers', 'PW':  'Prizewinners',
    'MY':  'Mystery',          'SF':  'Science Fiction', 'ROM': 'Romance',
    'YA':  'Young Adult',      'MID': 'Middlebrow',      'BIO': 'Biography',
    'MEM': 'Memoir',           'HIST':'History',         'MIX': 'Mixed Nonfiction',
}


# -------------------------------------------------------------------- helpers
def keynorm(x):
    """Normalize filenames so mojibake rows still join across the four files."""
    x = unicodedata.normalize('NFKD', str(x)).encode('ascii', 'ignore').decode()
    return re.sub(r'[^a-z0-9]', '', x.lower())


def repair_decimal_artifact(df, cols):
    """Divide extreme values by 1000 where a decimal separator was lost on load.

    A subset of rows in five metadata columns was ingested with the decimal
    mark dropped, inflating the value by exactly 1000x. Detected by IQR-based
    outlier test with a wide fence (6 x IQR), verified against the core
    distribution.
    """
    log = {}
    for c in cols:
        x = df[c]
        q1, q3 = x.quantile(.25), x.quantile(.75)
        iqr = q3 - q1
        bad = (x < q1 - 6 * iqr) | (x > q3 + 6 * iqr)
        df.loc[bad, c] = df.loc[bad, c] / 1000
        log[c] = int(bad.sum())
    return df, log


# =============================================================== 1. metadata
print('[1/6] loading metadata ...')
meta = pd.read_csv(UPL + 'CONLIT_META.csv', encoding='cp1252')
meta['BAYES_ranking'] = pd.to_numeric(meta['BAYES_ranking'], errors='coerce')
meta['_k'] = meta['ID'].map(keynorm)

meta, repair_log = repair_decimal_artifact(meta, ARTIFACT_COLS)
print('  repaired rows:', repair_log)

meta['Category'] = meta['Category'].str.strip()
meta['is_fic']   = (meta['Category'] == 'FIC').astype(int)


# =============================================================== 2. supersense
print('[2/6] loading supersense ...')
ss = pd.read_csv(UPL + 'CONLIT_SUPERSENSE.csv', encoding='cp1252')
ss['_k'] = ss['file_name'].map(keynorm)
ss_cols  = [c for c in ss.columns if c not in ('file_name', '_k')]

# normalize per 1,000 tokens (length-robust)
tok = meta.set_index('_k')['token_count']
ss = ss.set_index('_k')
ss_norm = ss[ss_cols].div(tok.reindex(ss.index), axis=0) * 1000
ss_norm.columns = [c + '_n' for c in ss_cols]
ss = pd.concat([ss[ss_cols], ss_norm], axis=1).reset_index()


# =============================================================== 3. LIWC
print('[3/6] loading LIWC ...')
liwc = pd.read_csv(UPL + 'CONLIT_LIWC.csv')
liwc['_k'] = liwc['Filename'].map(keynorm)
# drop identifiers + WC (WC ~ token_count at r = 0.996, not a style feature)
liwc_cols = [c for c in liwc.columns if c not in ('Filename', 'Segment', 'WC', '_k')]


# =============================================================== 4. unigrams -> foregrounding proxies
print('[4/6] loading unigrams and computing foregrounding proxies ...')
uni = pd.read_csv(UPL + 'CONLIT_UNIGRAM.csv')
uni['_k'] = uni['filename'].map(keynorm)
u_cols = [c for c in uni.columns if c not in ('filename', '_k')]

X = uni[u_cols].values.astype(np.float64)
row_sums = X.sum(axis=1)
Xn = X / row_sums[:, None]                    # per-work relative frequencies

# corpus-wide relative frequency of each word type (defines the norm)
corpus_freq = X.sum(axis=0) / X.sum()
log_freq    = np.log(corpus_freq + 1e-12)

fg = pd.DataFrame({'_k': uni['_k'].values})

# (1) lexical rarity: inverted mean log-frequency of words used
#     high = reaches for rarer vocabulary
fg['lex_rarity'] = -(Xn @ log_freq)

# (2) types-used rate: fraction of the top-10k vocabulary the work touches
fg['types_used_rate'] = (X > 0).sum(axis=1) / row_sums

# (3) hapax rate: fraction of tokens that occur exactly once in the work
fg['hapax_rate'] = (X == 1).sum(axis=1) / row_sums

# (4) rare-tail share: fraction of tokens in the 5,000 rarest word types
freq_rank = np.argsort(corpus_freq)           # ascending
rare_ids  = freq_rank[:5000]
fg['rare_tail_share'] = Xn[:, rare_ids].sum(axis=1)

# (5) Shannon entropy of the work's own distribution
fg['work_entropy'] = -(Xn * np.log(Xn + 1e-12)).sum(axis=1)

# (6) KL divergence from the corpus distribution
#     (Mukařovský's automatized background made explicit)
fg['KL_from_corpus'] = (
    Xn * (np.log(Xn + 1e-12) - np.log(corpus_freq + 1e-12))
).sum(axis=1)

# (7) KL from within-category centroid
#     (Shklovsky: estrangement against the convention of one's own form)
is_fic = meta.set_index('_k').reindex(uni['_k'].values)['is_fic'].values
fic_dist = Xn[is_fic == 1].mean(axis=0); fic_dist /= fic_dist.sum()
non_dist = Xn[is_fic == 0].mean(axis=0); non_dist /= non_dist.sum()
fg['KL_from_own_category'] = np.where(
    is_fic == 1,
    (Xn * (np.log(Xn + 1e-12) - np.log(fic_dist + 1e-12))).sum(axis=1),
    (Xn * (np.log(Xn + 1e-12) - np.log(non_dist + 1e-12))).sum(axis=1),
)

# (8) composite index: mean z-score of the four measures that pass the
#     within-fiction validation gate (lex_rarity, types_used_rate,
#     hapax_rate, rare_tail_share). See axes script for validation.
z = lambda s: (s - s.mean()) / s.std()
fg['foregrounding_index'] = (
    z(fg['lex_rarity']) + z(fg['types_used_rate'])
    + z(fg['hapax_rate']) + z(fg['rare_tail_share'])
) / 4


# =============================================================== 5. embodiment indices
print('[5/6] deriving embodiment indices ...')
emb = ss[['_k'] + [c + '_n' for c in EMBODIED + ABSTRACT]].copy()
emb['embodied_index'] = np.mean([z(emb[c + '_n']) for c in EMBODIED], axis=0)
emb['abstract_index'] = np.mean([z(emb[c + '_n']) for c in ABSTRACT], axis=0)
emb['poetic_index']   = emb['embodied_index'] - emb['abstract_index']
emb = emb[['_k', 'embodied_index', 'abstract_index', 'poetic_index']]


# =============================================================== 6. join everything
print('[6/6] joining and writing master CSV ...')
master = meta.merge(ss.drop(columns=ss_cols),   # keep only normalized supersenses
                    on='_k', how='inner')
master = master.merge(liwc[['_k'] + liwc_cols], on='_k', how='inner')
master = master.merge(fg,                       on='_k', how='inner')
master = master.merge(emb,                      on='_k', how='inner')
master = master.drop(columns=['_k'])            # drop the join key; keep original ID

# reorder columns for readability
id_cols = ['ID', 'Category', 'is_fic', 'Genre', 'Genre2', 'Pubdate',
           'Author_Last', 'Author_First', 'Work_Title',
           'Translation', 'PubHouse', 'Prize', 'WinnerShortlist',
           'Author_Gender', 'Author_Nationality', 'token_count',
           'goodreads_avg', 'total_ratings', 'goodreads_URL', 'Probability1P']
ss_feat_cols  = [c + '_n' for c in ss_cols]
derived_cols  = ['embodied_index', 'abstract_index', 'poetic_index']
fg_cols       = ['lex_rarity', 'types_used_rate', 'hapax_rate', 'rare_tail_share',
                 'work_entropy', 'KL_from_corpus', 'KL_from_own_category',
                 'foregrounding_index']

col_order = ([c for c in id_cols if c in master.columns]
             + META_FEATS + derived_cols + fg_cols + ss_feat_cols + liwc_cols)
master = master[col_order]

print(f'  master shape: {master.shape}')
print(f'  {master.Category.value_counts().to_dict()}')

master.to_csv(OUT + 'conlit_master.csv', index=False)
master.to_pickle('results/conlit_master.pkl')

# schema for downstream scripts
schema = {
    'meta_features':        META_FEATS,
    'supersense_features':  ss_feat_cols,
    'supersense_raw_names': ss_cols,
    'derived_semantic':     derived_cols,
    'foregrounding':        fg_cols,
    'liwc_features':        liwc_cols,
    'embodied_supersenses': [c + '_n' for c in EMBODIED],
    'abstract_supersenses': [c + '_n' for c in ABSTRACT],
    'artifact_repair_log':  repair_log,
    'n_works':              len(master),
    'n_fic':                int((master.is_fic == 1).sum()),
    'n_non':                int((master.is_fic == 0).sum()),
    'genre_label':          GENRE_LABEL,
}
with open('results/schema.json', 'w') as f:
    json.dump(schema, f, indent=2)

print(f'\n  wrote {OUT}conlit_master.csv  ({master.memory_usage(deep=True).sum()/1e6:.1f} MB)')
print('done.')
