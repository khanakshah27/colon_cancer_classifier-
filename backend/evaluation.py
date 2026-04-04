from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from collections import Counter

def evaluate_models(results, X, y):
    X_test = results["X_test"]
    y_test = results["y_test"]

    print("\n========== MODEL EVALUATION ==========\n")

    class_counts = Counter(y)
    cv_folds = max(2, min(5, min(class_counts.values())))
    print(f"Using {cv_folds}-fold Stratified Cross Validation\n")
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for name in ["svm", "rf"]:
        model, y_pred = results[name]
        print(f"\n----- {name.upper()} -----")
        print("Test Accuracy:", round(accuracy_score(y_test, y_pred), 4))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))
        cv_scores = cross_val_score(model, X, y, cv=skf)
        print("Cross-Validation Accuracy:", round(cv_scores.mean(), 4))

    print("\n=====================================\n")