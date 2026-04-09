const API = "http://127.0.0.1:5000";

const PLOT_TITLES = {
    pca:     "PCA — Normal vs Tumor",
    roc:     "ROC Curve (SVM)",
    volcano: "Volcano Plot — DEGs",
    heatmap: "Heatmap — Top 40 features",
    go:      "GO Enrichment — Biological Processes",
    kegg:    "KEGG Pathway Enrichment",
};

async function runPipeline() {
    const btn = document.getElementById("run-btn");
    const output = document.getElementById("pipeline-output");
    const errorDiv = document.getElementById("pipeline-error");

    btn.disabled = true;
    btn.querySelector(".btn-text").textContent = "Running (~2 min)...";
    btn.querySelector(".btn-icon").innerHTML = '<span class="spinner"></span>';
    output.classList.add("hidden");
    errorDiv.classList.add("hidden");

    try {
        const res  = await fetch(`${API}/run`);
        const data = await res.json();
        if (data.error) throw new Error(data.error);

        document.getElementById("svm-acc").textContent  = (data.svm_accuracy * 100).toFixed(1) + "%";
        document.getElementById("rf-acc").textContent   = (data.rf_accuracy  * 100).toFixed(1) + "%";
        document.getElementById("samples").textContent  = data.samples;
        document.getElementById("features").textContent = data.features.toLocaleString();
        document.getElementById("selected").textContent = data.selected_features;

        output.classList.remove("hidden");
        output.classList.add("fade-in");

    } catch (err) {
        errorDiv.textContent = err.message.includes("fetch") ? "Backend not running — start Flask first." : "Error: " + err.message;
        errorDiv.classList.remove("hidden");
    } finally {
        btn.disabled = false;
        btn.querySelector(".btn-text").textContent = "Run Analysis";
        btn.querySelector(".btn-icon").textContent = "→";
    }
}

async function loadAllPlots() {
    const btn      = document.getElementById("load-plots-btn");
    const output   = document.getElementById("plots-output");
    const grid     = document.getElementById("plots-grid");
    const errorDiv = document.getElementById("plots-error");

    btn.disabled = true;
    btn.querySelector(".btn-text").textContent = "Generating plots...";
    btn.querySelector(".btn-icon").innerHTML = '<span class="spinner"></span>';
    errorDiv.classList.add("hidden");
    grid.innerHTML = "";

    try {
        const res  = await fetch(`${API}/plots/all`);
        const data = await res.json();
        if (data.error) throw new Error(data.error);

        for (const [key, title] of Object.entries(PLOT_TITLES)) {
            if (!data[key]) continue;
            const box = document.createElement("div");
            box.className = "plot-box";
            box.innerHTML = `
                <div class="plot-title">${title}</div>
                <img src="${data[key]}" alt="${title}"/>
            `;
            grid.appendChild(box);
        }

        output.classList.remove("hidden");
        output.classList.add("fade-in");
        btn.querySelector(".btn-text").textContent = "Refresh Plots";

    } catch (err) {
        errorDiv.textContent = "Could not load plots: " + err.message;
        errorDiv.classList.remove("hidden");
        btn.querySelector(".btn-text").textContent = "Generate All Plots";
    } finally {
        btn.disabled = false;
        btn.querySelector(".btn-icon").textContent = "→";
    }
}

async function loadRandomSample() {
    const textarea = document.getElementById("sample-input");
    const errorDiv = document.getElementById("classify-error");
    textarea.value = "Loading...";
    errorDiv.classList.add("hidden");
    try {
        const res  = await fetch(`${API}/random_sample`);
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        textarea.value = data.values.join(", ");
    } catch (err) {
        textarea.value = "";
        errorDiv.textContent = "Could not load sample: " + err.message;
        errorDiv.classList.remove("hidden");
    }
}

function clearSample() {
    document.getElementById("sample-input").value = "";
    document.getElementById("classify-output").classList.add("hidden");
    document.getElementById("classify-error").classList.add("hidden");
}

async function classifySample() {
    const btn      = document.getElementById("classify-btn");
    const output   = document.getElementById("classify-output");
    const errorDiv = document.getElementById("classify-error");
    const raw      = document.getElementById("sample-input").value.trim();

    if (!raw) {
        errorDiv.textContent = "Please load a random sample or paste gene expression values.";
        errorDiv.classList.remove("hidden");
        return;
    }

    btn.disabled = true;
    btn.querySelector(".btn-text").textContent = "Classifying...";
    btn.querySelector(".btn-icon").innerHTML = '<span class="spinner"></span>';
    output.classList.add("hidden");
    errorDiv.classList.add("hidden");

    try {
        const values = raw.split(",").map(v => parseFloat(v.trim())).filter(v => !isNaN(v));
        const res    = await fetch(`${API}/classify`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ values })
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);

        const isTumor = data.prediction === 1;
        document.getElementById("result-box").className  = "result-box " + (isTumor ? "tumor" : "normal");
        document.getElementById("result-icon").textContent  = isTumor ? "◆" : "●";
        document.getElementById("result-label").textContent = isTumor ? "Tumor Detected" : "Normal Tissue";
        document.getElementById("result-desc").textContent  = isTumor
            ? "Gene expression consistent with primary colon adenocarcinoma"
            : "Gene expression consistent with normal colon mucosa";
        document.getElementById("result-conf").textContent  = `${(data.confidence * 100).toFixed(1)}% confidence`;

        output.classList.remove("hidden");
        output.classList.add("fade-in");

    } catch (err) {
        errorDiv.textContent = err.message.includes("fetch") ? "Backend not running." : "Error: " + err.message;
        errorDiv.classList.remove("hidden");
    } finally {
        btn.disabled = false;
        btn.querySelector(".btn-text").textContent = "Classify Sample";
        btn.querySelector(".btn-icon").textContent = "→";
    }
}
