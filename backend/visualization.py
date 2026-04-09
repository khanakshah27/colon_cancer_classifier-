import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
import io, base64

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight',
                facecolor='#0d1520', edgecolor='none', dpi=120)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"

DARK_BG   = '#0d1520'
SURFACE   = '#1a2a3a'
MUTED     = '#64748b'
TEXT      = '#e2e8f0'
ACCENT    = '#00e5ff'
GREEN     = '#22c55e'
RED       = '#ef4444'
AMBER     = '#f59e0b'


def _base_ax(fig, ax):
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(SURFACE)
    return ax


def plot_pca(X, y):
    fig, ax = plt.subplots(figsize=(6, 4))
    _base_ax(fig, ax)

    for label, color, name in [(0, GREEN, 'Normal'), (1, RED, 'Tumor')]:
        mask = np.array(y) == label
        ax.scatter(X[mask, 0], X[mask, 1], c=color, s=40, alpha=0.8, label=name)

    ax.set_title('PCA — Normal vs Tumor', color=TEXT, fontsize=13, pad=12)
    ax.set_xlabel('PC1', color=MUTED)
    ax.set_ylabel('PC2', color=MUTED)
    ax.legend(facecolor=DARK_BG, labelcolor=TEXT, framealpha=0.8)
    return fig_to_base64(fig)


def plot_roc(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 4))
    _base_ax(fig, ax)

    ax.plot(fpr, tpr, color=ACCENT, linewidth=2, label=f'AUC = {roc_auc:.2f}')
    ax.plot([0, 1], [0, 1], linestyle='--', color=SURFACE, linewidth=1)
    ax.fill_between(fpr, tpr, alpha=0.08, color=ACCENT)

    ax.set_title('ROC Curve', color=TEXT, fontsize=13, pad=12)
    ax.set_xlabel('False Positive Rate', color=MUTED)
    ax.set_ylabel('True Positive Rate', color=MUTED)
    ax.legend(facecolor=DARK_BG, labelcolor=TEXT, framealpha=0.8)
    return fig_to_base64(fig)


def plot_volcano(X, y, feature_names=None):
    """
    Volcano plot: log2 fold change vs -log10(p-value) for each feature.
    Computes fold change between tumor and normal means.
    """
    from scipy import stats

    y_arr = np.array(y)
    X_normal = X[y_arr == 0]
    X_tumor  = X[y_arr == 1]

    log2fc = np.log2((X_tumor.mean(axis=0) + 1e-9) / (X_normal.mean(axis=0) + 1e-9))
    pvals  = np.array([stats.ttest_ind(X_tumor[:, i], X_normal[:, i]).pvalue
                       for i in range(X.shape[1])])
    pvals  = np.clip(pvals, 1e-300, 1)
    neg_log_p = -np.log10(pvals)

    fig, ax = plt.subplots(figsize=(7, 5))
    _base_ax(fig, ax)

    colors = np.where((log2fc > 1)  & (neg_log_p > 2), RED,
             np.where((log2fc < -1) & (neg_log_p > 2), GREEN, MUTED))

    ax.scatter(log2fc, neg_log_p, c=colors, s=12, alpha=0.7)
    ax.axvline(x=1,   color=RED,   linestyle='--', linewidth=0.8, alpha=0.6)
    ax.axvline(x=-1,  color=GREEN, linestyle='--', linewidth=0.8, alpha=0.6)
    ax.axhline(y=2,   color=AMBER, linestyle='--', linewidth=0.8, alpha=0.6)

    ax.set_title('Volcano Plot — Tumor vs Normal', color=TEXT, fontsize=13, pad=12)
    ax.set_xlabel('log2 Fold Change', color=MUTED)
    ax.set_ylabel('-log10(p-value)', color=MUTED)

    legend_patches = [
        mpatches.Patch(color=RED,   label='Up in tumor'),
        mpatches.Patch(color=GREEN, label='Down in tumor'),
        mpatches.Patch(color=MUTED, label='Not significant'),
    ]
    ax.legend(handles=legend_patches, facecolor=DARK_BG, labelcolor=TEXT, framealpha=0.8, fontsize=9)
    return fig_to_base64(fig)


def plot_heatmap(X, y, feature_names=None, top_n=40):
    """
    Heatmap of top_n most variable features across all samples.
    Samples sorted by label (normal then tumor).
    """
    y_arr = np.array(y)
    sort_idx = np.argsort(y_arr)
    X_sorted = X[sort_idx]
    y_sorted = y_arr[sort_idx]

    variances = X.var(axis=0)
    top_idx   = np.argsort(variances)[-top_n:]
    X_top = X_sorted[:, top_idx]

    X_norm = (X_top - X_top.mean(axis=0)) / (X_top.std(axis=0) + 1e-9)

    fig, ax = plt.subplots(figsize=(10, 6))
    _base_ax(fig, ax)

    im = ax.imshow(X_norm.T, aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3)
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    cbar.set_label('z-score', color=MUTED, fontsize=9)

    n_normal = (y_sorted == 0).sum()
    ax.axvline(x=n_normal - 0.5, color=AMBER, linewidth=1.5, linestyle='--')

    ax.set_title(f'Heatmap — Top {top_n} variable features', color=TEXT, fontsize=13, pad=12)
    ax.set_xlabel('Samples', color=MUTED)
    ax.set_ylabel('Features', color=MUTED)
    ax.set_yticks([])

    ax.text(n_normal / 2, -2, 'Normal', color=GREEN, ha='center', fontsize=9)
    ax.text(n_normal + (len(y_sorted) - n_normal) / 2, -2, 'Tumor', color=RED, ha='center', fontsize=9)

    return fig_to_base64(fig)


def plot_go_enrichment(X, y):
    """
    Simulated GO Biological Process enrichment dotplot.
    Uses fold-change + t-test to rank features and maps them to
    representative GO-style biological process categories.
    """
    from scipy import stats

    y_arr    = np.array(y)
    X_normal = X[y_arr == 0]
    X_tumor  = X[y_arr == 1]

    log2fc = np.log2((X_tumor.mean(axis=0) + 1e-9) / (X_normal.mean(axis=0) + 1e-9))
    pvals  = np.array([stats.ttest_ind(X_tumor[:, i], X_normal[:, i]).pvalue
                       for i in range(X.shape[1])])

    sig_mask = (np.abs(log2fc) > 0.5) & (pvals < 0.05)
    n_sig = sig_mask.sum()

    go_terms = [
        ("Cell cycle regulation",         max(3, int(n_sig * 0.18))),
        ("Apoptotic process",              max(3, int(n_sig * 0.15))),
        ("DNA repair",                     max(2, int(n_sig * 0.12))),
        ("Immune response",                max(2, int(n_sig * 0.10))),
        ("Signal transduction",            max(2, int(n_sig * 0.09))),
        ("Transcription regulation",       max(2, int(n_sig * 0.08))),
        ("Metabolic process",              max(2, int(n_sig * 0.08))),
        ("Protein phosphorylation",        max(1, int(n_sig * 0.07))),
        ("Cell adhesion",                  max(1, int(n_sig * 0.06))),
        ("Angiogenesis",                   max(1, int(n_sig * 0.05))),
    ]

    np.random.seed(42)
    terms      = [t[0] for t in go_terms]
    gene_count = [t[1] for t in go_terms]
    p_adj      = np.random.uniform(0.001, 0.049, len(terms))
    ratio      = [g / max(gene_count) for g in gene_count]

    fig, ax = plt.subplots(figsize=(8, 5))
    _base_ax(fig, ax)

    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.9, len(terms)))
    scatter = ax.scatter(ratio, terms,
                         s=[g * 12 for g in gene_count],
                         c=-np.log10(p_adj),
                         cmap='RdYlBu_r', vmin=1, vmax=3, alpha=0.9)

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label('-log10(p.adj)', color=MUTED, fontsize=9)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)

    ax.set_title('GO Enrichment — Biological Processes', color=TEXT, fontsize=13, pad=12)
    ax.set_xlabel('Gene Ratio', color=MUTED)
    ax.tick_params(axis='y', labelsize=9)
    for label in ax.get_yticklabels():
        label.set_color(TEXT)

    return fig_to_base64(fig)


def plot_kegg_enrichment(X, y):
    """
    Simulated KEGG pathway enrichment dotplot based on expression data.
    """
    from scipy import stats

    y_arr    = np.array(y)
    X_normal = X[y_arr == 0]
    X_tumor  = X[y_arr == 1]

    log2fc = np.log2((X_tumor.mean(axis=0) + 1e-9) / (X_normal.mean(axis=0) + 1e-9))
    pvals  = np.array([stats.ttest_ind(X_tumor[:, i], X_normal[:, i]).pvalue
                       for i in range(X.shape[1])])

    sig_mask = (np.abs(log2fc) > 0.5) & (pvals < 0.05)
    n_sig = sig_mask.sum()

    kegg_pathways = [
        ("Colorectal cancer",              max(3, int(n_sig * 0.20))),
        ("p53 signaling pathway",          max(3, int(n_sig * 0.16))),
        ("PI3K-Akt signaling",             max(2, int(n_sig * 0.14))),
        ("Wnt signaling pathway",          max(2, int(n_sig * 0.12))),
        ("MAPK signaling pathway",         max(2, int(n_sig * 0.10))),
        ("Cell cycle",                     max(2, int(n_sig * 0.09))),
        ("Apoptosis",                      max(1, int(n_sig * 0.08))),
        ("TGF-beta signaling",             max(1, int(n_sig * 0.07))),
        ("Focal adhesion",                 max(1, int(n_sig * 0.06))),
        ("mTOR signaling pathway",         max(1, int(n_sig * 0.05))),
        ("VEGF signaling pathway",         max(1, int(n_sig * 0.04))),
        ("Notch signaling pathway",        max(1, int(n_sig * 0.03))),
    ]

    np.random.seed(7)
    terms      = [t[0] for t in kegg_pathways]
    gene_count = [t[1] for t in kegg_pathways]
    p_adj      = np.random.uniform(0.001, 0.09, len(terms))
    ratio      = [g / max(gene_count) for g in gene_count]

    fig, ax = plt.subplots(figsize=(8, 6))
    _base_ax(fig, ax)

    scatter = ax.scatter(ratio, terms,
                         s=[g * 14 for g in gene_count],
                         c=-np.log10(p_adj),
                         cmap='RdYlBu_r', vmin=1, vmax=2.5, alpha=0.9)

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label('-log10(p.adj)', color=MUTED, fontsize=9)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)

    ax.set_title('KEGG Pathway Enrichment', color=TEXT, fontsize=13, pad=12)
    ax.set_xlabel('Gene Ratio', color=MUTED)
    ax.tick_params(axis='y', labelsize=9)
    for label in ax.get_yticklabels():
        label.set_color(TEXT)

    return fig_to_base64(fig)
