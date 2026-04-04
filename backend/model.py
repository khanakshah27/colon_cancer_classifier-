import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut

def train_models(X, y, groups):
    """
    Uses Leave-One-Patient-Out CV: for each patient, train on all other
    patients and test on that patient's samples. This prevents the model
    from ever seeing a patient's normal sample when predicting their tumor,
    which is what caused 100% accuracy before.
    """
    logo = LeaveOneGroupOut()
    
    svm_preds = np.zeros(len(y))
    rf_preds  = np.zeros(len(y))
    svm_probs = np.zeros(len(y))

    fold_accuracies_svm = []
    fold_accuracies_rf  = []

    print(f"Running Leave-One-Patient-Out CV ({logo.get_n_splits(X, y, groups)} folds)...")

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

   
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        vt = VarianceThreshold(threshold=0.1)
        X_tr_v = vt.fit_transform(X_tr_s)
        X_te_v = vt.transform(X_te_s)

        k = min(20, X_tr_v.shape[1])
        sel = SelectKBest(score_func=f_classif, k=k)
        X_tr_f = sel.fit_transform(X_tr_v, y_tr)
        X_te_f = sel.transform(X_te_v)

        svm = SVC(kernel='linear', C=0.1, probability=True)
        svm.fit(X_tr_f, y_tr)
        svm_preds[test_idx] = svm.predict(X_te_f)
        svm_probs[test_idx] = svm.predict_proba(X_te_f)[:, 1]

        rf = RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=5, random_state=42)
        rf.fit(X_tr_f, y_tr)
        rf_preds[test_idx] = rf.predict(X_te_f)

        fold_accuracies_svm.append(accuracy_score(y_te, svm_preds[test_idx]))
        fold_accuracies_rf.append(accuracy_score(y_te, rf_preds[test_idx]))

    svm_acc = accuracy_score(y, svm_preds)
    rf_acc  = accuracy_score(y, rf_preds)

    print(f"LOPO SVM accuracy: {svm_acc:.4f}")
    print(f"LOPO RF  accuracy: {rf_acc:.4f}")

    scaler_final = StandardScaler()
    X_s = scaler_final.fit_transform(X)
    vt_final = VarianceThreshold(threshold=0.1)
    X_v = vt_final.fit_transform(X_s)
    k = min(20, X_v.shape[1])
    sel_final = SelectKBest(score_func=f_classif, k=k)
    X_f = sel_final.fit_transform(X_v, y)
    svm_final = SVC(kernel='linear', C=0.1, probability=True)
    svm_final.fit(X_f, y)

    return {
        "svm_acc":      svm_acc,
        "rf_acc":       rf_acc,
        "svm_preds":    svm_preds,
        "rf_preds":     rf_preds,
        "y":            y,
   
        "scaler":       scaler_final,
        "vt":           vt_final,
        "selector":     sel_final,
        "selected_idx": np.arange(k),
        "svm_final":    svm_final,
    }
