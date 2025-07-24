# ----------------------
# main.py (with SHAP, LIME, ELI5 explanations for all models)
# ----------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.data_loader import load_and_preprocess_data
from src.feature_selection import select_features
from src.models import get_models, get_param_grids, train_model
from src.evaluation import permutation_test
from src.xai import explain_with_shap, explain_with_lime, explain_with_eli5

import os
os.makedirs("outputs/shap_plots", exist_ok=True)
os.makedirs("outputs/lime", exist_ok=True)

# Load Data
data_path = "data/EEG_Data.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ File not found: {data_path}")

data = load_and_preprocess_data(data_path)
X = data.drop(columns=["specific.disorder"])
y = data["specific.disorder"]

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = get_models()
param_grids = get_param_grids()

results = []
best_models = {}
best_params = {}
all_selected_features = set()

for name, model in models.items():
    print(f"🔍 Training model: {name}")
    aucs, accs = [], []
    selected_features = None

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        print(f"  Fold {fold + 1}...")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        selected = select_features(X_train, y_train)
        selected_features = selected
        all_selected_features.update(selected)

        X_train_sel = X_train[selected]
        X_test_sel = X_test[selected]

        try:
            clf, params = train_model(model, param_grids[name], X_train_sel, y_train)
            y_pred_proba = clf.predict_proba(X_test_sel)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)

            aucs.append(roc_auc_score(y_test, y_pred_proba))
            accs.append(accuracy_score(y_test, y_pred))
        except Exception as e:
            print(f"    ⚠️ Fold failed: {e}")
            continue

    if clf:
        clf.fit(X[selected], y)
        p_val = permutation_test(clf, X[selected], y)

        results.append({
            "Model": name,
            "Mean AUC": np.mean(aucs) if aucs else np.nan,
            "Mean Accuracy": np.mean(accs) if accs else np.nan,
            "P-Value": p_val,
            "Best Params": params
        })

        best_models[name] = clf
        best_params[name] = params


# Save Results
results_df = pd.DataFrame(results)
results_df.to_csv("outputs/model_results.csv", index=False)

# Final Model
best_model_name = results_df.loc[results_df['Mean AUC'].idxmax(), 'Model']
best_model = best_models[best_model_name]
best_model.fit(X[selected_features], y)

