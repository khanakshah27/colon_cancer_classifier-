import GEOparse
import pandas as pd
import os
import re

def load_data(gse_id="GSE44076", destdir="./data"):

    if not os.path.exists(destdir):
        os.makedirs(destdir)

    print("Loading dataset...")
    gse = GEOparse.get_GEO(gse_id, destdir=destdir, silent=False)

    print("Extracting expression matrix...")
    gse_table = gse.pivot_samples('VALUE')
    data = gse_table.T

    print("Labeling samples...")
    labels = []
    patient_ids = []
    kept = []

    for sample in data.index:
        gsm = gse.gsms[sample]
        source = " ".join(gsm.metadata.get('source_name_ch1', [])).lower()
        chars  = " ".join(gsm.metadata.get('characteristics_ch1', [])).lower()

        match = re.search(r'individual id:\s*(\S+)', chars)
        patient_id = match.group(1) if match else "unknown"

        if source == "normal distant colon mucosa cells":
            labels.append(0)
            kept.append(sample)
            patient_ids.append(patient_id)
        elif source == "primary colon adenocarcinoma cells":
            labels.append(1)
            kept.append(sample)
            patient_ids.append(patient_id)


    data = data.loc[kept].copy()
    data['label'] = labels
    data['patient_id'] = patient_ids

    print(f"\nSamples: {len(kept)} (98 normal + 98 tumor, 98 patients)")
    print("Label distribution:")
    print(pd.Series(labels).value_counts())
    print(f"Unique patients: {len(set(patient_ids))}")

    return data
