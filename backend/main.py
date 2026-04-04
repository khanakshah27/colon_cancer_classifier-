from data_loader import load_data
from preprocessing import preprocess_data
from feature_selection import select_features
from model import train_models
from evaluation import evaluate_models
from visualization import plot_pca, plot_roc

def main():
    print("Loading data...")
    data = load_data()

    print("Preprocessing...")
    X_scaled, y, feature_names = preprocess_data(data)

    print("Feature selection...")
    X_selected, vt, selector, selected_idx = select_features(X_scaled, y)

    print("Training models...")
    results = train_models(X_selected, y)

    print("Evaluating...")
    evaluate_models(results, X_selected, y)

    print("Visualization...")
    plot_pca(X_selected, y)

    svm_model, _ = results["svm"]
    plot_roc(svm_model, results["X_test"], results["y_test"])

if __name__ == "__main__":
    main()