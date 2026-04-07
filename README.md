# 🧬 CancerScan — Colon Cancer Classifier

A full-stack machine learning web application that classifies colon tissue samples as **normal** or **cancerous (tumor)** using gene expression microarray data from the GEO public database.

---

## 📌 Project Overview

| Item | Detail |
|---|---|
| **Dataset** | GEO: [GSE44076](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE44076) |
| **Samples** | 196 total — 98 normal colon mucosa + 98 primary colon adenocarcinoma |
| **Patients** | 98 matched patients (each contributes 1 normal + 1 tumor sample) |
| **Raw Features** | ~49,000 microarray gene probes |
| **Selected Features** | Top 20 (after variance filtering + ANOVA F-test) |
| **Models** | SVM (linear kernel) + Random Forest |
| **Validation** | Leave-One-Patient-Out Cross Validation (LOPO-CV) |
| **Backend** | Python + Flask |
| **Frontend** | Vanilla HTML/CSS/JS |

---

## 🗂️ Project Structure

```
colon_cancer_classifier/
├── backend/
│   ├── app.py               # Flask REST API (main server)
│   ├── data_loader.py       # Downloads & parses GEO dataset (GSE44076)
│   ├── preprocessing.py     # Cleans data, handles missing values
│   ├── feature_selection.py # Standalone feature selection (used in main.py)
│   ├── model.py             # LOPO-CV training of SVM + Random Forest
│   ├── evaluation.py        # Evaluation utilities (used in main.py)
│   ├── visualization.py     # Generates PCA and ROC curve plots (base64 PNG)
│   ├── main.py              # Standalone CLI pipeline (non-web version)
│   └── requirements.txt     # Python dependencies
└── frontend/
    ├── index.html           # Main UI page
    ├── script.js            # API calls and DOM updates
    └── style.css            # Dark-themed styling (Syne + DM Mono fonts)
```

---

## ⚙️ How the Pipeline Works (Step by Step)

### 1. Data Loading — `data_loader.py`

- Uses the `GEOparse` library to **automatically download** dataset `GSE44076` from NCBI GEO.
- Extracts the **expression matrix** (samples × gene probes) using `pivot_samples('VALUE')`.
- Labels each sample by reading its `source_name_ch1` metadata field:
  - `"normal distant colon mucosa cells"` → label **0** (Normal)
  - `"primary colon adenocarcinoma cells"` → label **1** (Tumor)
- Extracts a **patient ID** from the `characteristics_ch1` field using a regex on `individual id:`.
- Keeps only matched samples (discards any samples without a valid label).

### 2. Preprocessing — `preprocessing.py`

- Drops non-feature columns (`label`, `patient_id`).
- Converts all values to numeric (coerces errors to `NaN`).
- **Imputes missing values** with the column mean.
- Returns `X` (feature matrix), `y` (labels), feature names, and patient IDs.

### 3. Feature Selection (inside `model.py` during CV)

Done **inside each CV fold** to prevent data leakage:

1. **VarianceThreshold** (`threshold=0.1`): Removes genes that barely vary across all samples — these carry no discriminating information.
2. **SelectKBest with f_classif** (`k=20`): Selects the top 20 genes based on ANOVA F-score between the two classes (normal vs. tumor).

### 4. Model Training — `model.py`

**Cross-Validation Strategy: Leave-One-Patient-Out (LOPO-CV)**

- For each of the 98 patients, train on the remaining 97 patients' samples (194 samples), and test on the held-out patient's 2 samples (1 normal + 1 tumor).
- This is crucial because each patient contributes *both* a normal and a tumor sample. Without patient-aware splitting, the model could learn patient-specific patterns and achieve artificially inflated accuracy.

Inside each fold:
1. `StandardScaler` normalizes features (fit on train, applied to test).
2. `VarianceThreshold` removes low-variance genes.
3. `SelectKBest` picks top 20 genes.
4. **SVM** (`kernel='linear'`, `C=0.1`, `probability=True`) is trained and predictions are collected.
5. **Random Forest** (`n_estimators=100`, `max_depth=3`, `min_samples_leaf=5`) is also trained.

After all folds, a **final model** is retrained on the entire dataset (for use in the `/classify` API endpoint).

### 5. Visualization — `visualization.py`

- **PCA Plot**: Projects the selected 20 features into 2 principal components and scatter-plots samples colored by class (green = normal, red = tumor).
- **ROC Curve**: Plots the SVM's true positive rate vs. false positive rate, with AUC displayed.
- Both plots are rendered with `matplotlib` on a dark background and returned as **base64-encoded PNG strings** (embedded directly in the API JSON response, no file I/O needed).

---

## 🌐 REST API — `app.py`

The Flask server exposes 4 endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check — returns `"Backend running"` |
| `GET` | `/run` | Runs the full ML pipeline (download → preprocess → train → evaluate). Returns accuracy metrics. |
| `GET` | `/plots` | Generates and returns PCA + ROC plots as base64 PNG. Must call `/run` first. |
| `GET` | `/random_sample` | Returns a random sample's raw feature values from the dataset. Used by the frontend to populate the classify form. |
| `POST` | `/classify` | Accepts `{ "values": [...] }` (raw feature array), runs it through the trained SVM pipeline, returns `prediction` (0 or 1) and `confidence` (probability). |

**Server state**: The app stores the trained models and dataset in a module-level `_state` dict (in-memory, lost on restart). The `/plots` and `/classify` endpoints depend on `/run` being called first.

---

## 🖥️ Frontend — `index.html` / `script.js` / `style.css`

The UI is a single dark-themed page with two main sections:

### Section 1: Run Full Analysis
- A **"Run Analysis"** button calls `GET /run`.
- While running, the button shows a spinner and disables (the pipeline takes ~1–2 minutes to download data and train).
- On success, displays: SVM Accuracy, Random Forest Accuracy, total samples, raw feature count, selected feature count.
- A **"Load Plots"** button then calls `GET /plots` and shows the PCA and ROC images.

### Section 2: Single Sample Prediction
- **"Load Random Sample"** calls `GET /random_sample` and fills a textarea with comma-separated gene values.
- **"Classify Sample"** calls `POST /classify` with the textarea values.
- Displays result: **"Tumor Detected"** (red) or **"Normal Tissue"** (green), with confidence percentage.

---

## 🚀 How to Run the Project

### Prerequisites
```bash
pip install GEOparse pandas numpy scikit-learn matplotlib seaborn xgboost flask flask-cors
```

### Start the Backend
```bash
cd backend
python app.py
# Server starts at http://127.0.0.1:5000
```

> ⚠️ On first run, `GEOparse` will download ~100MB of data from NCBI GEO. This is cached in `./data/` for future runs.

### Open the Frontend
Open `frontend/index.html` directly in a browser (no server needed for the frontend — it talks directly to the Flask backend at `localhost:5000`).

### CLI-only (no web)
```bash
cd backend
python main.py
```
This runs the full pipeline and prints results to the terminal.

---

## 🔑 Key Design Decisions & Why

### Why Leave-One-Patient-Out CV?
Each patient donates **both** a normal and a tumor sample. If a naive train/test split is used, the model can see a patient's normal tissue in training and then trivially recognize their tumor — achieving near-100% accuracy that doesn't generalize. LOPO-CV guarantees the model **never sees any sample from a patient during that patient's test fold**.

### Why Linear SVM with C=0.1?
Gene expression data is high-dimensional with relatively few samples. A linear kernel with small regularization (`C=0.1`) prevents overfitting, is fast to train, and is interpretable (the decision boundary is a hyperplane in gene space).

### Why SelectKBest (k=20) instead of using all ~49k genes?
Using all genes leads to the **curse of dimensionality** and overfitting with only ~196 samples. ANOVA F-test (`f_classif`) efficiently identifies the genes most statistically different between normal and tumor tissue, reducing noise.

### Why is feature selection done inside each CV fold?
To avoid **data leakage**. If you select features using the full dataset before splitting, the test-fold information influences which features were chosen, artificially inflating performance metrics.

---

## 📊 Expected Results

| Model | Accuracy (LOPO-CV) |
|---|---|
| SVM (linear, C=0.1) | ~85–95% |
| Random Forest (depth=3) | ~80–90% |

Actual results may vary slightly due to randomness in Random Forest.

---

## 📦 Dependencies (`requirements.txt`)

```
GEOparse       # Download and parse GEO datasets
pandas         # Data manipulation
numpy          # Numerical computing
scikit-learn   # ML models, preprocessing, metrics
matplotlib     # Plot generation
seaborn        # (imported but not directly used in current version)
xgboost        # (imported but not used in current version — available for extension)
```

> **Note**: `flask` and `flask-cors` are used by `app.py` but are missing from `requirements.txt`. Add them manually:
> ```bash
> pip install flask flask-cors
> ```

---

## 🧪 Dataset Details

- **GEO Accession**: GSE44076
- **Platform**: Microarray (Affymetrix or similar) — ~49,000 probe features
- **Tissue types**:
  - Label 0: Normal distant colon mucosa (healthy tissue)
  - Label 1: Primary colon adenocarcinoma (cancer tissue)
- **Design**: Matched pairs — each of the 98 patients contributes exactly one normal and one tumor sample

---

## 💡 Possible Improvements

- Add XGBoost classifier (already in `requirements.txt`)
- Persist trained model to disk (e.g., with `joblib`) so the pipeline doesn't re-run on every server restart
- Add cross-validation confidence intervals to the UI
- Display which top 20 genes were selected and their biological significance
- Add SHAP or feature importance explainability
- Add a loading progress bar (the ~2-minute wait has no server-side progress events currently)
