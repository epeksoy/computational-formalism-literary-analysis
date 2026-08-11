"""Chapter 4 pipeline — Step 3 of 5: univariate discrimination, PCA on the
semantic layer, prestige test within fiction, boundary cases, misclassification
by genre, length-confound check.

Outputs (in results folder):
  tab_univariate_metadata.csv
  tab_univariate_supersense.csv
  tab_univariate_LIWC.csv
  tab_univariate_foregrounding.csv
  tab_semantic_indices.csv
  tab_pca.csv
  tab_pca_pc1_loadings.csv
  tab_prestige_test.csv
  tab_genre_profiles.csv
  tab_errors_by_genre.csv
  tab_boundary_nonfic_as_fic.csv
  tab_boundary_fic_as_nonfic.csv
  tab_length_confound.csv
"""
import pandas as pd
import numpy as np
import json
import warnings

warnings.filterwarnings('ignore')

from scipy import stats
from sklearn.metrics import (roc_auc_score, confusion_matrix,
                             classification_report)
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

OUT = 'results/'
schema = json.load(open('results/schema.json'))
d = pd.read_pickle('results/conlit_scored.pkl').reset_index(drop=True)

MF = schema['meta_features']
NC = schema['supersense_features']
LC = schema['liwc_features']
FG = schema['foregrounding']
GENRE_LABEL = schema['genre_label']
y  = d['is_fic'].values


def univariate(cols, layer):
    """Cohen's d, direction-agnostic AUC, Mann-Whitney p, BH-corrected."""
    rows = []
    for c in cols:
        x = d[c]
        ok = x.notna()
        a, b = x[ok & (y == 1)], x[ok & (y == 0)]
        n1, n2 = len(a), len(b)
        sp = np.sqrt(
            ((n1 - 1) * a.std(ddof=1) ** 2 + (n2 - 1) * b.std(ddof=1) ** 2)
            / (n1 + n2 - 2)
        )
        dd = (a.mean() - b.mean()) / sp if sp > 0 else np.nan
        _, pt = stats.ttest_ind(a, b, equal_var=False)
        _, pu = stats.mannwhitneyu(a, b, alternative='two-sided')
        auc_raw = roc_auc_score(y[ok], x[ok])
        rows.append({
            'layer': layer, 'feature': c,
            'FIC_mean': a.mean(), 'FIC_sd': a.std(ddof=1),
            'NON_mean': b.mean(), 'NON_sd': b.std(ddof=1),
            'cohens_d': dd, 'direction': 'FIC' if dd > 0 else 'NON',
            'AUC': max(auc_raw, 1 - auc_raw),
            'p_welch': pt, 'p_mw': pu, 'n': n1 + n2,
        })
    out = pd.DataFrame(rows)
    out['p_BH'] = multipletests(out['p_mw'], method='fdr_bh')[1]
    return out.sort_values('AUC', ascending=False)


# ============================================================ 1. univariate
print('[1] univariate discrimination ...')
tab_meta = univariate(MF, 'metadata')
tab_ss   = univariate(NC, 'supersense')
tab_liwc = univariate(LC, 'LIWC')
tab_fg   = univariate(FG, 'foregrounding')

for name, tab in [('metadata',      tab_meta),
                  ('supersense',    tab_ss),
                  ('LIWC',          tab_liwc),
                  ('foregrounding', tab_fg)]:
    tab.to_csv(OUT + f'tab_univariate_{name}.csv', index=False)
    n_sig = (tab['p_BH'] < .05).sum()
    n_big = (tab['cohens_d'].abs() > 0.8).sum()
    print(f'  {name:14s}  n = {len(tab):3d}  BH-significant = {n_sig}   |d| > 0.8 = {n_big}')


# ============================================================ 2. semantic indices summary
print('\n[2] semantic composition indices ...')
rows = []
for idx in ['embodied_index', 'abstract_index', 'poetic_index']:
    a, b = d.loc[y == 1, idx], d.loc[y == 0, idx]
    sp = np.sqrt(
        ((len(a) - 1) * a.std() ** 2 + (len(b) - 1) * b.std() ** 2)
        / (len(a) + len(b) - 2)
    )
    dd = (a.mean() - b.mean()) / sp
    auc_raw = roc_auc_score(y, d[idx])
    rows.append({'index': idx,
                 'FIC_mean': a.mean(), 'FIC_sd': a.std(),
                 'NON_mean': b.mean(), 'NON_sd': b.std(),
                 'cohens_d': dd,
                 'AUC': max(auc_raw, 1 - auc_raw)})
sc = pd.DataFrame(rows)
sc.to_csv(OUT + 'tab_semantic_indices.csv', index=False)
print(sc.to_string(index=False, float_format=lambda v: f'{v:+.3f}'))


# ============================================================ 3. PCA on supersense (unsupervised!)
print('\n[3] PCA on supersense ...')
X = StandardScaler().fit_transform(SimpleImputer(strategy='median').fit_transform(d[NC]))
p = PCA(n_components=5).fit(X)
S = p.transform(X)

rows = []
for i in range(5):
    auc_pc = roc_auc_score(y, S[:, i])
    rows.append({'component': f'PC{i+1}',
                 'variance_explained': p.explained_variance_ratio_[i],
                 'AUC_fic_non': max(auc_pc, 1 - auc_pc)})
    print(f'  PC{i+1}: {p.explained_variance_ratio_[i]*100:5.2f}% var'
          f'   AUC(FIC/NON) = {max(auc_pc, 1-auc_pc):.3f}')

pd.DataFrame(rows).to_csv(OUT + 'tab_pca.csv', index=False)

# PC1 loadings for interpretation
load1 = pd.Series(p.components_[0], index=NC).sort_values()
load1.to_csv(OUT + 'tab_pca_pc1_loadings.csv', header=['loading'])

for i in range(3):
    d[f'PC{i+1}'] = S[:, i]


# ============================================================ 4. prestige test within fiction
print('\n[4] prestige test within fiction ...')
fic = d[d.is_fic == 1]
PW  = fic[fic.Genre == 'PW']
GEN = fic[fic.Genre.isin(['ROM', 'MY', 'SF', 'YA', 'BS'])]
print(f'  PW n = {len(PW)}   Genre fiction n = {len(GEN)}')

rows = []
for idx in ['embodied_index', 'abstract_index', 'poetic_index',
            'fictionality_score', 'prestige_score', 'foregrounding_score',
            'p_fiction_combined']:
    a, b = PW[idx], GEN[idx]
    sp = np.sqrt(
        ((len(a) - 1) * a.std() ** 2 + (len(b) - 1) * b.std() ** 2)
        / (len(a) + len(b) - 2)
    )
    dd = (a.mean() - b.mean()) / sp
    _, pv = stats.ttest_ind(a, b, equal_var=False)
    auc_raw = roc_auc_score(
        np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b])
    rows.append({'measure': idx,
                 'PW_mean': a.mean(), 'Genre_mean': b.mean(),
                 'cohens_d': dd, 'p': pv,
                 'AUC': max(auc_raw, 1 - auc_raw)})
    print(f'  {idx:22s}  PW = {a.mean():+.3f}  Genre = {b.mean():+.3f}'
          f'  d = {dd:+.2f}  AUC = {max(auc_raw, 1-auc_raw):.3f}')

pd.DataFrame(rows).to_csv(OUT + 'tab_prestige_test.csv', index=False)


# ============================================================ 5. genre profiles
print('\n[5] genre profiles ...')
gg = d.groupby('Genre').agg(
    n=('is_fic', 'size'),
    category=('Category', 'first'),
    embodied=('embodied_index', 'mean'),
    abstract=('abstract_index', 'mean'),
    poetic=('poetic_index', 'mean'),
    fictionality=('fictionality_score', 'mean'),
    prestige=('prestige_score', 'mean'),
    foregrounding=('foregrounding_score', 'mean'),
    p_fic_combined=('p_fiction_combined', 'mean'),
    bayes=('BAYES_ranking', 'mean'),
    lex_rarity=('lex_rarity', 'mean'),
    hapax_rate=('hapax_rate', 'mean'),
).reset_index()
gg['label'] = gg['Genre'].map(GENRE_LABEL)
gg = gg.sort_values('poetic', ascending=False)
gg.to_csv(OUT + 'tab_genre_profiles.csv', index=False)
print(gg[['Genre', 'label', 'category', 'n',
          'embodied', 'abstract', 'poetic',
          'foregrounding', 'prestige', 'p_fic_combined']].to_string(
    index=False, float_format=lambda v: f'{v:+.3f}'))


# ============================================================ 6. confusion + errors by genre
print('\n[6] confusion matrix and errors by genre ...')
d['pred'] = (d['p_fiction_combined'] >= .5).astype(int)
cm = confusion_matrix(y, d['pred'])
print(pd.DataFrame(cm, index=['true NON', 'true FIC'],
                       columns=['pred NON', 'pred FIC']))

err = d.assign(err=(d.pred != d.is_fic).astype(int)).groupby('Genre').agg(
    n=('err', 'size'), errors=('err', 'sum'),
    category=('Category', 'first')).reset_index()
err['rate']  = err['errors'] / err['n']
err['label'] = err['Genre'].map(GENRE_LABEL)
err = err.sort_values('rate', ascending=False)
err.to_csv(OUT + 'tab_errors_by_genre.csv', index=False)
print()
print(err[['Genre', 'label', 'category', 'n', 'errors', 'rate']].to_string(
    index=False, float_format=lambda v: f'{v:.3f}'))


# ============================================================ 7. boundary cases
print('\n[7] boundary cases ...')
b_nf = d[d.is_fic == 0].nlargest(25, 'p_fiction_combined')[
    ['ID', 'Author_Last', 'Work_Title', 'Genre', 'Pubdate',
     'p_fiction_combined', 'p_fiction_ss', 'p_fiction_liwc',
     'embodied_index', 'abstract_index', 'poetic_index',
     'foregrounding_score', 'prestige_score']]
b_nf.to_csv(OUT + 'tab_boundary_nonfic_as_fic.csv', index=False)

b_fn = d[d.is_fic == 1].nsmallest(25, 'p_fiction_combined')[
    ['ID', 'Author_Last', 'Work_Title', 'Genre', 'Pubdate',
     'p_fiction_combined', 'p_fiction_ss', 'p_fiction_liwc',
     'embodied_index', 'abstract_index', 'poetic_index',
     'foregrounding_score', 'prestige_score']]
b_fn.to_csv(OUT + 'tab_boundary_fic_as_nonfic.csv', index=False)


# ============================================================ 8. length confound
print('\n[8] length confound ...')
rows = []
for c in MF + NC + FG:
    ok = d[c].notna()
    r, _ = stats.pearsonr(d.loc[ok, c], d.loc[ok, 'token_count'])
    if   c in MF: layer = 'metadata'
    elif c in NC: layer = 'supersense'
    else:         layer = 'foregrounding'
    rows.append({'feature': c, 'layer': layer, 'r_with_token_count': r})

lc = pd.DataFrame(rows)
lc.to_csv(OUT + 'tab_length_confound.csv', index=False)
for lay in ['metadata', 'supersense', 'foregrounding']:
    v = lc[lc.layer == lay]['r_with_token_count'].abs()
    print(f'  {lay:14s}  mean |r| = {v.mean():.3f}   max |r| = {v.max():.3f}')

d.to_pickle('results/conlit_scored.pkl')
print('\ndone.')
