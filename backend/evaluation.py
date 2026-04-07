from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
import numpy as np

def evaluate_models(results):
    y_true = results["y"]
    svm_preds = results["svm_preds"]
    rf_preds  = results["rf_preds"]

    print("\n========== MODEL EVALUATION (Leave-One-Patient-Out) ==========\n")

    for name, preds in [("SVM", svm_preds), ("Random Forest", rf_preds)]:
        print(f"----- {name} -----")
        print(f"Accuracy:          {accuracy_score(y_true, preds):.4f}")
        print(f"ROC-AUC:           {roc_auc_score(y_true, preds):.4f}")
        print(f"Confusion Matrix:\n{confusion_matrix(y_true, preds)}")
        print(f"Classification Report:\n{classification_report(y_true, preds, target_names=['Normal','Tumor'])}")

    print("==============================================================\n")
