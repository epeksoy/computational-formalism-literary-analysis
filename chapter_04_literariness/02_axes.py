"""Chapter 4 pipeline — Step 2 of 5: fit the three axes and test orthogonality.

Fits three classifiers, saves out-of-fold decision-function scores on every
work, and tests whether the axes are geometrically independent.

  fictionality axis  LIWC classifier: FIC vs NON             (n = 2,752)
  prestige axis      LIWC classifier: PW vs genre fiction    (n = 1,349 in training)
  foregrounding      composite of four unigram measures       (already in master)

Also runs the model comparison table (feature blocks x AUC).

Outputs:
  tab_models.csv     model comparison across feature blocks
  conlit_axes.csv    ID, category, genre, three axis scores, three P(fiction)
  out_axes.json      cosines, bootstrap CIs, correlations
"""
import pandas as pd
import numpy as np
import json
import warnings

warnings.filterwarnings('ignore')

from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     cross_val_score)
from sklearn.metrics import roc_auc_score

OUT = 'results/'

schema = json.load(open('results/schema.json'))
d = pd.read_pickle('results/conlit_master.pkl').reset_index(drop=True)

MF = schema['meta_features']
NC = schema['supersense_features']
LC = schema['liwc_features']
FG = schema['foregrounding']
y  = d['is_fic'].values

cv = StratifiedKFold(10, shuffle=True, random_state=42)


def pipe(clf=None):
    """Median-imputed, standardized logistic regression by default."""
    return Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sc',  StandardScaler()),
        ('clf', LogisticRegression(max_iter=5000) if clf is None else clf),
    ])


# ============================================================ 1. MODEL COMPARISON
STYLO = ['tuldava_score', 'avg_sentence_length', 'avg_word_length']
NARR  = ['event_count', 'circuitousness', 'speed_avg', 'speed_min',
         'volume', 'protagonist_concentration', 'total_characters']
RECEP = ['BAYES_ranking']

BLOCKS = [
    ('A. Reception only (BAYES)',            RECEP),
    ('B. Stylometric only (3)',              STYLO),
    ('C. Narrative only (7)',                NARR),
    ('D. Metadata layer (11)',               MF),
    ('E. Supersense layer (41)',             NC),
    ('F. LIWC layer (117)',                  LC),
    ('G. Foregrounding proxies (8)',         FG),
    ('H. Metadata + Supersense',             MF + NC),
    ('I. Metadata + Supersense + LIWC',      MF + NC + LC),
    ('J. All layers (all features)',         MF + NC + LC + FG),
]

rows = []
print('[1] model comparison ...')
for name, cols in BLOCKS:
    r = cross_val_score(pipe(), d[cols].values, y, cv=cv, scoring='roc_auc',  n_jobs=-1)
    a = cross_val_score(pipe(), d[cols].values, y, cv=cv, scoring='accuracy', n_jobs=-1)
    rows.append({'model': name, 'k': len(cols),
                 'AUC': r.mean(), 'AUC_sd': r.std(),
                 'Acc': a.mean(), 'Acc_sd': a.std()})
    print(f'  {name:42s} k={len(cols):4d}  AUC = {r.mean():.4f} ± {r.std():.4f}')

# random forest ceiling check
rf = cross_val_score(
    pipe(RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)),
    d[MF + NC + LC].values, y, cv=cv, scoring='roc_auc', n_jobs=-1)
rows.append({'model': 'K. RandomForest ceiling (169)',
             'k': len(MF + NC + LC),
             'AUC': rf.mean(), 'AUC_sd': rf.std(),
             'Acc': np.nan, 'Acc_sd': np.nan})
print(f'  K. RandomForest ceiling on 169 features           AUC = {rf.mean():.4f}')

pd.DataFrame(rows).to_csv(OUT + 'tab_models.csv', index=False)


# ============================================================ 2. FICTIONALITY AXIS (LIWC)
print('\n[2] fictionality axis (LIWC) ...')
d['fictionality_score'] = cross_val_predict(
    pipe(), d[LC].values, y, cv=cv, method='decision_function')
d['p_fiction_liwc'] = cross_val_predict(
    pipe(), d[LC].values, y, cv=cv, method='predict_proba')[:, 1]

# supersense-based fictionality (for the convergence argument in the outline)
d['p_fiction_ss'] = cross_val_predict(
    pipe(), d[NC].values, y, cv=cv, method='predict_proba')[:, 1]

# combined-model P(fiction) — the one to quote for boundary cases
d['p_fiction_combined'] = cross_val_predict(
    pipe(), d[MF + NC + LC].values, y, cv=cv, method='predict_proba')[:, 1]

fic_auc = roc_auc_score(y, d['fictionality_score'])
print(f'  fictionality LIWC axis   AUC = {fic_auc:.4f}')


# ============================================================ 3. PRESTIGE AXIS (within fiction)
print('\n[3] prestige axis (LIWC within fiction) ...')
GENRE_FIC = ['ROM', 'MY', 'SF', 'YA', 'BS']
in_train  = (d['is_fic'] == 1) & d['Genre'].isin(['PW'] + GENRE_FIC)
Xt = d.loc[in_train, LC].values
yt = (d.loc[in_train, 'Genre'] == 'PW').astype(int).values
print(f'  training on n = {in_train.sum()}   (PW = {yt.sum()}, genre = {len(yt) - yt.sum()})')

# out-of-fold prestige score on the training subset
prestige_oof = np.full(len(d), np.nan)
prestige_oof[in_train.values] = cross_val_predict(
    pipe(), Xt, yt, cv=cv, method='decision_function')

# fit on the whole subset and score everyone (works outside training get the fit score)
m_prestige = pipe().fit(Xt, yt)
d['prestige_score'] = m_prestige.decision_function(d[LC].values)
d.loc[in_train.values, 'prestige_score'] = prestige_oof[in_train.values]

sub_auc = roc_auc_score(yt, prestige_oof[in_train.values])
print(f'  prestige axis            AUC (PW vs genre, OOF) = {sub_auc:.4f}')


# ============================================================ 4. FOREGROUNDING AXIS
# Already computed in step 1; alias for consistency.
d['foregrounding_score'] = d['foregrounding_index']


# ============================================================ 5. ORTHOGONALITY
print('\n[5] orthogonality between axes ...')


def cosine_of_coefs(cols, mask_A, y_A, mask_B, y_B, C=1.0, boots=200):
    """Cosine between the two logistic coefficient vectors, with bootstrap."""
    p = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sc',  StandardScaler()),
        ('clf', LogisticRegression(max_iter=5000, C=C)),
    ])
    wA = p.fit(d.loc[mask_A, cols].values, y_A).named_steps['clf'].coef_[0]
    wB = p.fit(d.loc[mask_B, cols].values, y_B).named_steps['clf'].coef_[0]
    obs = float(np.dot(wA, wB) / (np.linalg.norm(wA) * np.linalg.norm(wB)))

    rng = np.random.default_rng(42)
    A_idx = np.where(mask_A.values)[0]
    B_idx = np.where(mask_B.values)[0]
    cs = []
    for _ in range(boots):
        ia = rng.choice(A_idx, len(A_idx), replace=True)
        ib = rng.choice(B_idx, len(B_idx), replace=True)
        if len(np.unique(y_B[np.searchsorted(B_idx, ib)])) < 2:
            continue
        wa = p.fit(d.iloc[ia][cols].values,
                   y_A[np.searchsorted(A_idx, ia)]).named_steps['clf'].coef_[0]
        wb = p.fit(d.iloc[ib][cols].values,
                   y_B[np.searchsorted(B_idx, ib)]).named_steps['clf'].coef_[0]
        cs.append(np.dot(wa, wb) / (np.linalg.norm(wa) * np.linalg.norm(wb)))
    return obs, np.array(cs)


# fictionality axis: all works
mask_A = pd.Series([True] * len(d))
y_A    = y
# prestige axis: fiction only, PW vs genre fiction
mask_B = in_train
y_B    = yt

obs_ax, boots_ax = cosine_of_coefs(LC, mask_A, y_A, mask_B, y_B)
print(f'  cosine(fictionality LIWC, prestige LIWC) = {obs_ax:+.3f}'
      f'   angle = {np.degrees(np.arccos(obs_ax)):.1f}°')
print(f'  bootstrap 95% CI: [{np.percentile(boots_ax, 2.5):+.3f}, '
      f'{np.percentile(boots_ax, 97.5):+.3f}]')

# score-level correlations across all works
r_fic_pre, p_fic_pre = stats.pearsonr(d['fictionality_score'], d['prestige_score'])
r_fic_fg,  p_fic_fg  = stats.pearsonr(d['fictionality_score'], d['foregrounding_score'])
r_pre_fg,  p_pre_fg  = stats.pearsonr(d['prestige_score'],     d['foregrounding_score'])

# within fiction
fic = d[d.is_fic == 1]
r_pf_wf, p_pf_wf = stats.pearsonr(fic.prestige_score,     fic.foregrounding_score)
r_ff_wf, p_ff_wf = stats.pearsonr(fic.fictionality_score, fic.foregrounding_score)

axes_res = {
    'cosine_fictionality_prestige_LIWC': obs_ax,
    'cosine_bootstrap_95CI': [float(np.percentile(boots_ax, 2.5)),
                              float(np.percentile(boots_ax, 97.5))],
    'angle_deg': float(np.degrees(np.arccos(obs_ax))),
    'r_fic_pre_all':          r_fic_pre,
    'r_fic_fg_all':           r_fic_fg,
    'r_pre_fg_all':           r_pre_fg,
    'r_pre_fg_within_fiction': r_pf_wf,
    'r_fic_fg_within_fiction': r_ff_wf,
    'AUC_fictionality_LIWC':          float(fic_auc),
    'AUC_prestige_PW_vs_genre_LIWC':  float(sub_auc),
}
with open(OUT + 'out_axes.json', 'w') as f:
    json.dump(axes_res, f, indent=2)

print('\nscore-level correlations:')
print(f'  r(fictionality, prestige)     all works       = {r_fic_pre:+.3f}')
print(f'  r(fictionality, foregrounding) all works      = {r_fic_fg:+.3f}')
print(f'  r(prestige, foregrounding)    all works       = {r_pre_fg:+.3f}')
print(f'  r(prestige, foregrounding)    within fiction  = {r_pf_wf:+.3f}')
print(f'  r(fictionality, foregrounding) within fiction = {r_ff_wf:+.3f}')


# ============================================================ 6. write outputs
axis_cols = ['ID', 'Category', 'is_fic', 'Genre', 'Pubdate',
             'Author_Last', 'Work_Title', 'token_count', 'BAYES_ranking',
             'embodied_index', 'abstract_index', 'poetic_index',
             'fictionality_score', 'prestige_score', 'foregrounding_score',
             'p_fiction_ss', 'p_fiction_liwc', 'p_fiction_combined']
d[axis_cols].to_csv(OUT + 'conlit_axes.csv', index=False)

# update the master CSV to include axis scores
master = pd.read_csv(OUT + 'conlit_master.csv')
for c in ['fictionality_score', 'prestige_score', 'foregrounding_score',
          'p_fiction_ss', 'p_fiction_liwc', 'p_fiction_combined']:
    master[c] = d[c].values
master.to_csv(OUT + 'conlit_master.csv', index=False)

d.to_pickle('results/conlit_scored.pkl')
print(f'\n  wrote {OUT}tab_models.csv, {OUT}conlit_axes.csv, {OUT}out_axes.json')
print('done.')
