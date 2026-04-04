import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression

def select_features(X_train, y_train, X_test):
    # Step 1: Remove zero/near-zero variance (fit on train only)
    vt = VarianceThreshold(threshold=0.1)
    X_train_var = vt.fit_transform(X_train)
    X_test_var  = vt.transform(X_test)

    # Step 2: SelectKBest — k=20 to keep it tight on 147 train samples
    k = min(20, X_train_var.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_sel = selector.fit_transform(X_train_var, y_train)
    X_test_sel  = selector.transform(X_test_var)

    return X_train_sel, X_test_sel, vt, selector, np.arange(k)