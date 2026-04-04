const API = "http://127.0.0.1:5000";

async function runPipeline() {
    const btn = document.getElementById("run-btn");
    const output = document.getElementById("pipeline-output");
    const errorDiv = document.getElementById("pipeline-error");

    btn.disabled = true;
    btn.querySelector(".btn-text").textContent = "Running...";
    btn.querySelector(".btn-icon").innerHTML = '<span class="spinner"></span>';
    output.classList.add("hidden");
    errorDiv.classList.add("hidden");

    try {
        const res = await fetch(`${API}/run`);
        const data = await res.json();

        if (data.error) throw new Error(data.error);

        document.getElementById("svm-acc").textContent  = (data.svm_accuracy * 100).toFixed(1) + "%";
        document.getElementById("rf-acc").textContent   = (data.rf_accuracy * 100).toFixed(1) + "%";
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

async function loadRandomSample() {
    const textarea = document.getElementById("sample-input");
    textarea.value = "Loading...";
    try {
        const res = await fetch(`${API}/random_sample`);
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        textarea.value = data.values.join(", ");
    } catch (err) {
        textarea.value = "";
        const errorDiv = document.getElementById("classify-error");
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
    const btn = document.getElementById("classify-btn");
    const output = document.getElementById("classify-output");
    const errorDiv = document.getElementById("classify-error");
    const raw = document.getElementById("sample-input").value.trim();

    if (!raw) {
        errorDiv.textContent = "Please paste gene expression values or load a random sample.";
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
        const res = await fetch(`${API}/classify`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ values })
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);

        const box = document.getElementById("result-box");
        const icon = document.getElementById("result-icon");
        const label = document.getElementById("result-label");
        const conf = document.getElementById("result-conf");

        const isTumor = data.prediction === 1;
        box.className = "result-box " + (isTumor ? "tumor" : "normal");
        icon.textContent = isTumor ? "◆" : "●";
        label.textContent = isTumor ? "Tumor Detected" : "Normal Tissue";
        conf.textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}% · Model: SVM`;

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