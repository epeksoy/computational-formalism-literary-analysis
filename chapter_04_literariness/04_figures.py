"""Chapter 4 pipeline — Step 4 of 5: figures.

Reads the scored master data and result tables, writes 14 publication figures
to results/
  effect_metadata.png             univariate effect sizes, metadata layer
  effect_supersense.png           univariate effect sizes, supersense layer
  effect_liwc.png                 univariate effect sizes, LIWC layer (top 30)
  model_ablation.png              layer ablation, cross-validated AUC per block
  roc_curves.png                  ROC curves by feature layer
  pca_unsupervised.png            PC1 × PC2 scatter — recovering the divide unsupervised
  orthogonal_axes.png             fictionality axis × prestige axis (chapter centerpiece)
  foregrounding_by_genre.png      composite foregrounding by genre + convergence within fiction
  genre_map_semantic.png          genres on embodied × abstract axes
  errors_by_genre.png             misclassification rate per genre
  prestige_semantic_test.png      the failed prestige test on the semantic axis
  axis_correlation_matrix.png     3x3 correlation matrix of the three axes
  length_confound.png             |r| with token_count by feature layer
  feature_distributions.png       boxplots of strongest discriminator per layer
"""
import pandas as pd
import numpy as np
import json
import warnings
import matplotlib

warnings.filterwarnings('ignore')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import roc_curve, auc

plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['DejaVu Serif'],
    'font.size':        9,
    'axes.linewidth':   .8,
    'axes.edgecolor':   '#333333',
    'axes.labelsize':   9,
    'axes.titlesize':   10,
    'axes.titleweight': 'bold',
    'xtick.labelsize':  8,
    'ytick.labelsize':  8,
    'legend.fontsize':  8,
    'figure.dpi':       160,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
    'axes.grid':        True,
    'grid.alpha':       .25,
    'grid.linewidth':   .5,
    'axes.axisbelow':   True,
})

FIC_C, NON_C, PW_C = '#1f4e79', '#b45309', '#8b0000'
OUT = 'results/'

schema      = json.load(open('results/schema.json'))
GENRE_LABEL = schema['genre_label']
d           = pd.read_pickle('results/conlit_scored.pkl')

tab_meta = pd.read_csv(OUT + 'tab_univariate_metadata.csv')
tab_ss   = pd.read_csv(OUT + 'tab_univariate_supersense.csv')
tab_liwc = pd.read_csv(OUT + 'tab_univariate_LIWC.csv')
tab_mod  = pd.read_csv(OUT + 'tab_models.csv')
tab_gen  = pd.read_csv(OUT + 'tab_genre_profiles.csv')
tab_err  = pd.read_csv(OUT + 'tab_errors_by_genre.csv')
axes_res = json.load(open(OUT + 'out_axes.json'))


# -------------------------------------------------------------- metadata effect sizes
fig, ax = plt.subplots(figsize=(7, 4.2))
t = tab_meta.sort_values('cohens_d')
cols = [FIC_C if v > 0 else NON_C for v in t.cohens_d]
ax.barh(t.feature, t.cohens_d, color=cols, edgecolor='black', linewidth=.5, height=.7)
ax.axvline(0, color='black', lw=.9)
for x in (-0.8, -0.5, 0.5, 0.8):
    ax.axvline(x, color='grey', ls=':', lw=.7)
ax.set_xlabel("Cohen's $d$   (negative = higher in nonfiction, positive = higher in fiction)")
ax.set_title('Metadata layer: effect sizes for the FIC/NON contrast')
ax.legend(handles=[Patch(facecolor=FIC_C, edgecolor='k', label='Higher in fiction'),
                   Patch(facecolor=NON_C, edgecolor='k', label='Higher in nonfiction')],
          loc='lower right')
for i, (f, v) in enumerate(zip(t.feature, t.cohens_d)):
    ax.text(v + (.08 if v > 0 else -.08), i, f'{v:+.2f}',
            va='center', ha='left' if v > 0 else 'right', fontsize=7.5)
ax.set_xlim(-3.1, 2.0)
plt.savefig(OUT + 'effect_metadata.png'); plt.close()


# -------------------------------------------------------------- supersense effect sizes
fig, ax = plt.subplots(figsize=(7, 7))
t = tab_ss.sort_values('cohens_d')
t['lab'] = t['feature'].str.replace('_n', '', regex=False)
cols = [FIC_C if v > 0 else NON_C for v in t.cohens_d]
ax.barh(t.lab, t.cohens_d, color=cols, edgecolor='black', linewidth=.4, height=.75)
ax.axvline(0, color='black', lw=.9)
for x in (-0.8, 0.8):
    ax.axvline(x, color='grey', ls=':', lw=.7)
ax.set_xlabel("Cohen's $d$")
ax.set_title('Semantic composition: all 41 supersenses, normalized per 1,000 tokens')
ax.legend(handles=[Patch(facecolor=FIC_C, edgecolor='k', label='Higher in fiction'),
                   Patch(facecolor=NON_C, edgecolor='k', label='Higher in nonfiction')],
          loc='lower right')
plt.savefig(OUT + 'effect_supersense.png'); plt.close()


# -------------------------------------------------------------- LIWC effect sizes (top 30)
fig, ax = plt.subplots(figsize=(7, 7))
t = tab_liwc.reindex(tab_liwc.cohens_d.abs().sort_values(ascending=False).index).head(30)
t = t.sort_values('cohens_d')
theory = {'shehe', 'they', 'i', 'we', 'you', 'ppron', 'ipron', 'focuspast',
          'focuspresent', 'focusfuture', 'Perception', 'visual', 'auditory',
          'feeling', 'motion', 'space', 'attention', 'Analytic', 'WPS', 'BigWords'}
cols = [FIC_C if v > 0 else NON_C for v in t.cohens_d]
ax.barh(t.feature, t.cohens_d, color=cols, edgecolor='black', linewidth=.4, height=.72)
ax.axvline(0, color='black', lw=.9)
ax.set_xlabel("Cohen's $d$")
ax.set_title("LIWC: top 30 discriminating features\n"
             "(italicized labels are theory-relevant — Cohn's signposts and Hamburger's tense)")
for i, f in enumerate(t.feature):
    if f in theory:
        ax.get_yticklabels()[i].set_style('italic')
        ax.get_yticklabels()[i].set_weight('bold')
ax.legend(handles=[Patch(facecolor=FIC_C, edgecolor='k', label='Higher in fiction'),
                   Patch(facecolor=NON_C, edgecolor='k', label='Higher in nonfiction')],
          loc='lower right')
plt.savefig(OUT + 'effect_liwc.png'); plt.close()


# -------------------------------------------------------------- model ablation
fig, ax = plt.subplots(figsize=(7.4, 4.2))
t = tab_mod[~tab_mod.model.str.startswith('K')].copy()
t['short'] = t.model.str.replace(r'^[A-J]\. ', '', regex=True)
t = t.iloc[::-1]
cl = [PW_C if 'Reception' in m else FIC_C for m in t.short]
ax.barh(t.short, t.AUC, xerr=t.AUC_sd, color=cl, edgecolor='black',
        linewidth=.5, height=.66, error_kw=dict(lw=.9, capsize=3))
ax.axvline(.5, color='grey', ls='--', lw=1)
ax.text(.505, -.6, 'chance', fontsize=7.5, color='grey')
ax.set_xlim(.45, 1.02)
ax.set_xlabel('Cross-validated AUC (10-fold, ±1 SD)')
ax.set_title('Layer ablation: what carries the fiction/nonfiction contrast')
for i, (v, s) in enumerate(zip(t.AUC, t.AUC_sd)):
    ax.text(v + s + .008, i, f'{v:.3f}', va='center', fontsize=8)
plt.savefig(OUT + 'model_ablation.png'); plt.close()


# -------------------------------------------------------------- ROC curves
fig, ax = plt.subplots(figsize=(5.6, 5.2))
y_true = d.is_fic.values
for col, lab, c, ls in [
    ('p_fiction_combined', 'Combined (metadata + SS + LIWC)', '#1f4e79', '-'),
    ('p_fiction_liwc',     'LIWC only (117)',                  '#2e8b57', '--'),
    ('p_fiction_ss',       'Supersense only (41)',             '#b45309', '-.'),
]:
    fpr, tpr, _ = roc_curve(y_true, d[col])
    ax.plot(fpr, tpr, color=c, ls=ls, lw=1.8,
            label=f'{lab}  AUC = {auc(fpr, tpr):.3f}')

sub = d.dropna(subset=['BAYES_ranking'])
fpr, tpr, _ = roc_curve(sub.is_fic, sub.BAYES_ranking)
ax.plot(fpr, tpr, color=PW_C, ls=':', lw=1.8,
        label=f'Reception only (BAYES)  AUC = {auc(fpr, tpr):.3f}')
ax.plot([0, 1], [0, 1], color='grey', lw=.9, label='Chance  AUC = 0.500')

ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate')
ax.set_title('ROC curves by feature layer\n(out-of-fold, 10-fold CV)')
ax.legend(loc='lower right')
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
plt.savefig(OUT + 'roc_curves.png'); plt.close()


# -------------------------------------------------------------- unsupervised PCA
fig, ax = plt.subplots(figsize=(6.4, 5.4))
for lab, c, mk in [(1, FIC_C, 'o'), (0, NON_C, '^')]:
    s = d[d.is_fic == lab]
    ax.scatter(s.PC1, s.PC2, s=9, c=c, marker=mk, alpha=.45, linewidths=0,
               label='Fiction' if lab else 'Nonfiction')
ax.set_xlabel('PC1  (29.5% of variance; AUC = 0.967)')
ax.set_ylabel('PC2  (13.6% of variance; AUC = 0.512)')
ax.set_title('Unsupervised PCA of semantic composition\n'
             'PC1 recovers the fiction/nonfiction divide before any supervision')
ax.legend(markerscale=2.2, frameon=True)
plt.savefig(OUT + 'pca_unsupervised.png'); plt.close()


# -------------------------------------------------------------- THE ORTHOGONALITY FIGURE
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
ax = axes[0]
for lab, c, mk in [(1, FIC_C, 'o'), (0, NON_C, '^')]:
    s = d[d.is_fic == lab]
    ax.scatter(s.fictionality_score, s.prestige_score, s=7, c=c, marker=mk,
               alpha=.35, linewidths=0,
               label='Fiction' if lab else 'Nonfiction')
ax.axvline(0, color='k', lw=.8, ls='--'); ax.axhline(0, color='k', lw=.8, ls=':')
ax.set_xlabel('Fictionality axis  (LIWC discriminant, FIC vs NON)')
ax.set_ylabel('Prestige axis  (LIWC discriminant, PW vs genre fiction)')
ci_lo, ci_hi = axes_res['cosine_bootstrap_95CI']
ax.set_title(f'The two axes are orthogonal\n'
             f'cosine = {axes_res["cosine_fictionality_prestige_LIWC"]:+.3f}, '
             f'angle = {axes_res["angle_deg"]:.1f}°, '
             f'bootstrap 95% CI [{ci_lo:+.2f}, {ci_hi:+.2f}]', fontsize=9.5)
ax.legend(markerscale=2.5, loc='lower left')

ax = axes[1]
g = tab_gen.copy()
for _, r in g.iterrows():
    c = FIC_C if r.category == 'FIC' else NON_C
    ax.scatter(r.fictionality, r.prestige, s=r.n * .8, c=c, alpha=.55,
               edgecolors='black', linewidths=.7, zorder=3)
    ax.annotate(r.label, (r.fictionality, r.prestige),
                fontsize=8, xytext=(0, -14), textcoords='offset points',
                ha='center', zorder=4)
ax.axvline(0, color='k', lw=.8, ls='--'); ax.axhline(0, color='k', lw=.8, ls=':')
ax.set_xlabel('Fictionality axis'); ax.set_ylabel('Prestige axis')
ax.set_title('Genre centroids in the two-axis space\n'
             'Fictionality and prestige vary independently', fontsize=9.5)
ax.legend(handles=[Patch(facecolor=FIC_C, edgecolor='k', alpha=.55, label='Fiction'),
                   Patch(facecolor=NON_C, edgecolor='k', alpha=.55, label='Nonfiction')],
          loc='lower left')
plt.tight_layout(); plt.savefig(OUT + 'orthogonal_axes.png'); plt.close()


# -------------------------------------------------------------- foregrounding
fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4))
ax = axes[0]
g = tab_gen.copy().sort_values('foregrounding', ascending=True)
cl = [FIC_C if c == 'FIC' else NON_C for c in g.category]
ax.barh(g.label, g.foregrounding, color=cl, edgecolor='black',
        linewidth=.5, height=.72)
ax.axvline(0, color='k', lw=.9)
ax.set_xlabel('Composite foregrounding index (mean $z$, higher = more distinctive vocabulary)')
ax.set_title('Nonfiction is more lexically foregrounded than fiction\n'
             'Within fiction, prizewinners rise to the top', fontsize=10)
ax.legend(handles=[Patch(facecolor=FIC_C, edgecolor='k', label='Fiction'),
                   Patch(facecolor=NON_C, edgecolor='k', label='Nonfiction')],
          loc='lower right')

ax = axes[1]
fic = d[d.is_fic == 1]
markers = [
    ('PW',  PW_C,      'o', 40, 'Prizewinners'),
    ('NYT', '#1f4e79', '^', 20, 'NYT'),
    ('MID', '#4b6cb7', 's', 20, 'Middlebrow'),
    ('BS',  '#8fa1c9', 'v', 20, 'Bestsellers'),
    ('MY',  '#d99400', 'D', 18, 'Mystery'),
    ('SF',  '#2e8b57', 'P', 20, 'Sci-Fi'),
    ('ROM', '#e08d95', 'p', 20, 'Romance'),
    ('YA',  '#7a6ba1', 'h', 20, 'Young Adult'),
]
for g_code, c, m, sz, lab in markers:
    s = fic[fic.Genre == g_code]
    ax.scatter(s.prestige_score, s.foregrounding_score, s=sz, c=c, marker=m,
               alpha=.55, edgecolors='none', label=lab)
ax.axvline(0, color='grey', ls=':', lw=.7); ax.axhline(0, color='grey', ls=':', lw=.7)
ax.set_xlabel('Prestige axis  (higher = closer to prizewinners)')
ax.set_ylabel('Composite foregrounding index')
ax.set_title(f'Within fiction, foregrounding and prestige converge\n'
             f'r = {axes_res["r_pre_fg_within_fiction"]:+.2f}, p < .001', fontsize=10)
ax.legend(loc='upper left', ncol=2, fontsize=7.5, markerscale=1.1)
plt.tight_layout(); plt.savefig(OUT + 'foregrounding_by_genre.png'); plt.close()


# -------------------------------------------------------------- genre semantic map
fig, ax = plt.subplots(figsize=(7, 5.6))
for _, r in tab_gen.iterrows():
    c = FIC_C if r.category == 'FIC' else NON_C
    ax.scatter(r.embodied, r.abstract, s=r.n * .75, c=c, alpha=.55,
               edgecolors='black', linewidths=.7, zorder=3)
    ax.annotate(r.label, (r.embodied, r.abstract), fontsize=8,
                xytext=(0, -14), textcoords='offset points',
                ha='center', zorder=4)
ax.axhline(0, color='grey', lw=.7, ls=':')
ax.axvline(0, color='grey', lw=.7, ls=':')
ax.set_xlabel('Embodied index  (body, perception, contact, motion)')
ax.set_ylabel('Abstract index  (act, group, cognition, relation, state)')
ax.set_title('Genre map of semantic composition\n(marker area proportional to n)')
ax.legend(handles=[Patch(facecolor=FIC_C, edgecolor='k', alpha=.55, label='Fiction'),
                   Patch(facecolor=NON_C, edgecolor='k', alpha=.55, label='Nonfiction')],
          loc='upper right')
plt.savefig(OUT + 'genre_map_semantic.png'); plt.close()


# -------------------------------------------------------------- errors by genre
fig, ax = plt.subplots(figsize=(7, 4))
t = tab_err.sort_values('rate')
cl = [FIC_C if c == 'FIC' else NON_C for c in t.category]
ax.barh(t.label, t.rate * 100, color=cl, edgecolor='black', linewidth=.5, height=.7)
ax.set_xlabel('Misclassification rate (%)')
ax.set_title('Where the classifier fails: error rate by genre\n'
             'Residuals concentrate in memoir and mixed nonfiction')
for i, (v, n, e) in enumerate(zip(t.rate, t.n, t.errors)):
    ax.text(v * 100 + .25, i, f'{e}/{n}', va='center', fontsize=7.5)
ax.legend(handles=[Patch(facecolor=FIC_C, edgecolor='k', label='Fiction'),
                   Patch(facecolor=NON_C, edgecolor='k', label='Nonfiction')],
          loc='lower right')
ax.set_xlim(0, 13)
plt.savefig(OUT + 'errors_by_genre.png'); plt.close()


# -------------------------------------------------------------- prestige test on semantic axis (negative result)
fig, ax = plt.subplots(figsize=(6.6, 4.4))
fic = d[d.is_fic == 1]
grp = [('PW',  'Prizewinners'), ('NYT', 'NYT'), ('MID', 'Middlebrow'),
       ('BS',  'Bestsellers'),  ('SF',  'Sci-Fi'), ('MY',  'Mystery'),
       ('YA',  'Young Adult'),  ('ROM', 'Romance')]
data = [fic.loc[fic.Genre == g, 'poetic_index'].dropna() for g, _ in grp]
bp = ax.boxplot(data, patch_artist=True, widths=.6, showfliers=False,
                medianprops=dict(color='black', lw=1.3),
                labels=[l for _, l in grp])
for p in bp['boxes']:
    p.set_facecolor(FIC_C); p.set_alpha(.6)
    p.set_edgecolor('black'); p.set_linewidth(.6)
ax.set_ylabel('Poetic index  (embodied − abstract)')
ax.set_title('Within fiction, the poetic index does not track prestige\n'
             'Prizewinners rank mid-pack; romance scores highest ($d$ = −0.13)')
plt.setp(ax.get_xticklabels(), rotation=35, ha='right')
ax.grid(axis='x', visible=False)
plt.savefig(OUT + 'prestige_semantic_test.png'); plt.close()


# -------------------------------------------------------------- three-axis correlations
fig, ax = plt.subplots(figsize=(4.4, 4.4))
cols   = ['fictionality_score', 'prestige_score', 'foregrounding_score']
labels = ['Fictionality',       'Prestige',        'Foregrounding']
C = d[cols].corr().values
im = ax.imshow(C, cmap='RdBu_r', vmin=-1, vmax=1)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f'{C[i, j]:+.2f}', ha='center', va='center',
                fontsize=11,
                color='white' if abs(C[i, j]) > .5 else 'black',
                weight='bold' if i == j else 'normal')
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(labels, rotation=25, ha='right')
ax.set_yticklabels(labels)
ax.set_title('Correlations among the three axes\n(all 2,752 works)', pad=10)
plt.colorbar(im, ax=ax, shrink=.7)
plt.savefig(OUT + 'axis_correlation_matrix.png'); plt.close()


# -------------------------------------------------------------- length confound
fig, ax = plt.subplots(figsize=(6.2, 4))
lc = pd.read_csv(OUT + 'tab_length_confound.csv')
color_map = {'metadata': '#b45309', 'supersense': '#2e8b57', 'foregrounding': '#1f4e79'}
pos_map   = {'metadata': -.28,      'supersense': 0,          'foregrounding': .28}

rng = np.random.default_rng(3)
for lay in ['metadata', 'supersense', 'foregrounding']:
    v = lc[lc.layer == lay]['r_with_token_count'].abs()
    off = pos_map[lay]
    ax.scatter(np.full(len(v), off) + rng.normal(0, .05, len(v)),
               v, s=22, c=color_map[lay], alpha=.65,
               edgecolors='black', linewidths=.4,
               label=f'{lay} (mean |r| = {v.mean():.3f})')

ax.axhline(.3, color='grey', ls='--', lw=.9)
ax.text(.4, .31, '|r| = 0.3', fontsize=7.5, color='grey')
ax.set_xticks(list(pos_map.values()))
ax.set_xticklabels(['Metadata', 'Supersense', 'Foregrounding'])
ax.set_ylabel('|r| with token_count')
ax.set_title('Length-confound check by feature layer')
ax.legend(loc='upper right')
ax.set_xlim(-.55, .55)
plt.savefig(OUT + 'length_confound.png'); plt.close()


# -------------------------------------------------------------- feature distributions
key = ['tuldava_score',   'avg_sentence_length',
       'noun.body_n',     'noun.act_n',
       'visual',          'Analytic',
       'lex_rarity',      'hapax_rate']
titles = ['Tuldava score (metadata)',      'Avg sentence length (metadata)',
          'noun.body per 1k (supersense)', 'noun.act per 1k (supersense)',
          'visual (LIWC)',                 'Analytic (LIWC)',
          'Lexical rarity (foregrounding)','Hapax rate (foregrounding)']

fig, axes = plt.subplots(2, 4, figsize=(12.5, 5.8))
for ax, c, ti in zip(axes.ravel(), key, titles):
    data = [d.loc[d.is_fic == 1, c].dropna(),
            d.loc[d.is_fic == 0, c].dropna()]
    bp = ax.boxplot(data, patch_artist=True, widths=.55, showfliers=False,
                    medianprops=dict(color='black', lw=1.4),
                    labels=['FIC', 'NON'])
    for p, cl in zip(bp['boxes'], [FIC_C, NON_C]):
        p.set_facecolor(cl); p.set_alpha(.75)
        p.set_edgecolor('black'); p.set_linewidth(.6)
    ax.set_title(ti, fontsize=9)
    ax.grid(axis='x', visible=False)

fig.suptitle('Strongest discriminators from each feature layer',
             fontsize=11, fontweight='bold', y=1.01)
plt.tight_layout(); plt.savefig(OUT + 'feature_distributions.png'); plt.close()

print('wrote 14 figures (numbered names removed) to', OUT)
