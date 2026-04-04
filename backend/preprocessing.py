import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
 
    patient_ids = data['patient_id'] if 'patient_id' in data.columns else None
    
    numeric = data.drop(columns=[c for c in ['label', 'patient_id'] if c in data.columns])
    numeric = numeric.apply(pd.to_numeric, errors='coerce')
    numeric = numeric.fillna(numeric.mean())

    X = numeric.values
    y = data['label']

    return X, y, numeric.columns, patient_ids
