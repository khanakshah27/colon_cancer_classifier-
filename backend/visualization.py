import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
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

def plot_pca(X, y):
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('#0d1520')
    ax.set_facecolor('#0d1520')

    colors = ['#22c55e' if label == 0 else '#ef4444' for label in y]
    scatter = ax.scatter([], [], c=[], s=40, alpha=0.8) 
    ax.scatter(
        [x for x, l in zip(X[:, 0], y) if l == 0],
        [x for x, l in zip(X[:, 1], y) if l == 0],
        c='#22c55e', s=40, alpha=0.8, label='Normal'
    )
    ax.scatter(
        [x for x, l in zip(X[:, 0], y) if l == 1],
        [x for x, l in zip(X[:, 1], y) if l == 1],
        c='#ef4444', s=40, alpha=0.8, label='Tumor'
    )

    ax.set_title('PCA — Colon Cancer vs Normal', color='white', fontsize=13, pad=12)
    ax.set_xlabel('PC1', color='#64748b')
    ax.set_ylabel('PC2', color='#64748b')
    ax.tick_params(colors='#64748b')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1a2a3a')
    ax.legend(facecolor='#0d1520', labelcolor='white', framealpha=0.8)

    return fig_to_base64(fig)

def plot_roc(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('#0d1520')
    ax.set_facecolor('#0d1520')

    ax.plot(fpr, tpr, color='#00e5ff', linewidth=2, label=f'AUC = {roc_auc:.2f}')
    ax.plot([0, 1], [0, 1], linestyle='--', color='#1a2a3a', linewidth=1)
    ax.fill_between(fpr, tpr, alpha=0.08, color='#00e5ff')

    ax.set_title('ROC Curve', color='white', fontsize=13, pad=12)
    ax.set_xlabel('False Positive Rate', color='#64748b')
    ax.set_ylabel('True Positive Rate', color='#64748b')
    ax.tick_params(colors='#64748b')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1a2a3a')
    ax.legend(facecolor='#0d1520', labelcolor='white', framealpha=0.8)

    return fig_to_base64(fig)
