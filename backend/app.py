from flask import Flask, jsonify, request
from flask_cors import CORS
from data_loader import load_data
from preprocessing import preprocess_data
from model import train_models
from evaluation import evaluate_models
from visualization import plot_pca, plot_roc, plot_volcano, plot_heatmap, plot_go_enrichment, plot_kegg_enrichment
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

_state = {}

@app.route("/")
def home():
    return "Backend running"

@app.route("/run", methods=["GET"])
def run_pipeline_api():
    try:
        data = load_data()
        X, y, feature_names, patient_ids = preprocess_data(data)
        groups = pd.factorize(patient_ids)[0]
        results = train_models(X, y, groups)
        evaluate_models(results)

        _state["results"]      = results
        _state["X_raw"]        = X
        _state["y"]            = y
        _state["feature_names"] = feature_names

        return jsonify({
            "svm_accuracy":      round(float(results["svm_acc"]), 4),
            "rf_accuracy":       round(float(results["rf_acc"]), 4),
            "samples":           int(len(X)),
            "features":          int(X.shape[1]),
            "selected_features": int(results["selected_idx"].shape[0]),
            "cv_method":         "Leave-One-Patient-Out"
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)})


@app.route("/plots", methods=["GET"])
def get_plots():
    try:
        if "results" not in _state:
            return jsonify({"error": "Run the pipeline first."})

        r  = _state["results"]
        X  = _state["X_raw"]
        y  = _state["y"]

        X_s   = r["scaler"].transform(X)
        X_v   = r["vt"].transform(X_s)
        X_f   = r["selector"].transform(X_v)

        pca     = PCA(n_components=2)
        X_pca   = pca.fit_transform(X_f)

        return jsonify({
            "pca": plot_pca(X_pca, y),
            "roc": plot_roc(r["svm_final"], X_f, y),
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)})


@app.route("/plots/all", methods=["GET"])
def get_all_plots():
    try:
        if "results" not in _state:
            return jsonify({"error": "Run the pipeline first."})

        r  = _state["results"]
        X  = _state["X_raw"]
        y  = _state["y"]
        fn = _state.get("feature_names")

        X_s = r["scaler"].transform(X)
        X_v = r["vt"].transform(X_s)
        X_f = r["selector"].transform(X_v)

        pca   = PCA(n_components=2)
        X_pca = pca.fit_transform(X_f)

        return jsonify({
            "pca":      plot_pca(X_pca, y),
            "roc":      plot_roc(r["svm_final"], X_f, y),
            "volcano":  plot_volcano(X_f, y, fn),
            "heatmap":  plot_heatmap(X_f, y, fn),
            "go":       plot_go_enrichment(X_f, y),
            "kegg":     plot_kegg_enrichment(X_f, y),
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)})


@app.route("/random_sample", methods=["GET"])
def random_sample():
    try:
        if "X_raw" not in _state:
            return jsonify({"error": "Run the pipeline first."})
        X   = _state["X_raw"]
        idx = np.random.randint(0, len(X))
        return jsonify({"values": X[idx].tolist(), "index": int(idx)})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/classify", methods=["POST"])
def classify():
    try:
        if "results" not in _state:
            return jsonify({"error": "Run the pipeline first."})

        body   = request.get_json()
        values = np.array(body["values"], dtype=float).reshape(1, -1)
        r      = _state["results"]

        v_s = r["scaler"].transform(values)
        v_v = r["vt"].transform(v_s)
        v_f = r["selector"].transform(v_v)

        pred = int(r["svm_final"].predict(v_f)[0])
        prob = float(r["svm_final"].predict_proba(v_f)[0][pred])

        return jsonify({"prediction": pred, "confidence": round(prob, 4)})

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
