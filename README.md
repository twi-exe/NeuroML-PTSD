# **EEG Biomarkers for PTSD Diagnosis using Explainable AI (XAI)**

*An interpretable machine learning framework for identifying neurobiological markers of PTSD using EEG.*

> This repository implements a clinically transparent and computationally efficient approach for PTSD diagnosis using EEG, Machine Learning, and Explainable AI.
> Visual + methodological details are based on the research poster included in the repo.

---

## **Overview**

Post-Traumatic Stress Disorder (PTSD) is currently diagnosed primarily through clinical interviews, which can be subjective and vary across clinicians. EEG provides a non-invasive, scalable, and cost-effective way to capture neural dynamics, but raw EEG is high-dimensional and difficult to interpret.

This project tackles these challenges by:

* Reducing high-dimensional EEG data to the most relevant features using **ElasticNet**
* Training multiple ML classifiers on these features
* Applying **Explainable AI (XAI)** tools to highlight interpretable neural biomarkers
* Identifying EEG frequency-band and connectivity patterns linked to PTSD

---

## **Dataset**

| Property       | Description                                                                            |
| -------------- | -------------------------------------------------------------------------------------- |
| Total Subjects | **104** (52 PTSD, 52 Healthy)                                                          |
| Data Type      | qEEG Resting-State                                                                     |
| Features       | **1140** PSD & Functional Connectivity scores across 19 channels and 6 frequency bands |
| Preprocessing  | Missing-value imputation + categorical encoding                                        |

---

## **Methodology**

1. **Preprocessing**

   * Remove irrelevant fields
   * Impute missing values
   * Encode categorical variables

2. **Feature Selection**

   * ElasticNet regression reduces ~**97%** of features
   * Retains only clinically meaningful and interpretable EEG features

   #### ElasticNet Feature Weights
   The ElasticNet regression model was used to identify the most relevant EEG features by balancing L1 and L2 regularization. This approach ensures sparsity while retaining correlated features.
   ![ElasticNet Weights](results/elasticnet_weights.png)

3. **Model Training**

   * ML models tested:
     `SVC`, `KNN`, `RandomForest`, `XGBoost`, `LGBM`, `AdaBoost`, `CatBoost`
   * Hyperparameter tuning performed using **Optuna**

   #### Model Feature Importance
   Feature importance was derived from the best-performing models to understand the contribution of each feature to PTSD classification.
   ![Model Feature Importance](results/model_feature_importance.png)

4. **Explainable AI**

   * **SHAP** → Global interpretability (biomarker patterns)
   * **LIME** → Individual patient-level interpretation
   * **ELI5** → Feature ranking & verification

   #### SHAP Summary Plot
   SHAP (SHapley Additive exPlanations) was used to visualize the global impact of features on model predictions. This plot highlights the most influential features across the dataset.
   ![SHAP Summary Plot](results/shap_summary_plot.png)

   #### Permutation Importance
   Permutation importance was calculated to validate the robustness of feature rankings and their impact on model performance.
   ![Permutation Importance](results/permutation_importance.png)

---

## **Best Model Performance**

| Model                            | AUC Score | Notes                                            |
| -------------------------------- | --------- | ------------------------------------------------ |
| **Random Forest (Optuna-Tuned)** | **0.85**  | Most stable & interpretable                      |
| CatBoost                         | 0.836     | Competitive performance                          |
| XGBoost                          | 0.821     | Robust but slightly less interpretable           |
| LGBM                             | 0.810     | Efficient but sensitive to hyperparameters       |

> Permutation testing confirmed significance at **p < 0.05**.

---

## **Key Findings**

* **Beta Power in Right Primary Motor Cortex (C4 region)** emerged as a strong marker.
* **Alpha band functional connectivity** between frontal and parietal regions showed meaningful group differences.
* **IQ scores** contributed significantly and interactively to PTSD classification.
* PTSD patients show:

  * Elevated **Beta / High-Beta** (hyperarousal signature)
  * Reduced **Delta / Alpha** (attenuated cognitive & emotional regulation)

---

## **Repository Structure**

```
├── .git/                   # Git version control metadata
├── .gitignore              # Git ignore rules
├── .vscode/                # VS Code workspace settings
├── README.md               # Project documentation
├── catboost_info/          # CatBoost training logs
│   ├── catboost_training.json
│   ├── learn/
│   ├── learn_error.tsv
│   └── time_left.tsv
├── data/                   # EEG dataset (not included due to privacy)
│   └── EEG_data.csv
├── main.py                 # Entry point for the project
├── notebooks/              # Jupyter notebooks for analysis
│   ├── prelim.ipynb
│   ├── visuals.ipynb
│   └── xai.ipynb
├── pipeline/               # Core pipeline scripts
│   ├── config.py
│   ├── data_loader.py
│   ├── evaluation.py
│   ├── feature_selection.py
│   ├── model_registry.py
│   ├── train_and_evaluate.py
│   └── xai.py
├── requirements.txt        # Python dependencies
├── results/                # Model outputs, SHAP plots, etc.
│   ├── bayes_results.csv
│   ├── elasticnet_weights.png
│   ├── final_auc.txt
│   ├── final_summary.csv
│   ├── model_feature_importance.png
│   ├── optuna_results.csv
│   ├── permutation_importance.png
│   ├── selected_features.txt
│   ├── shap_summary_plot.png
│   └── summary.png
├── src/                    # Source code for experiments
│   ├── bayes_optimization.py
│   ├── config.py
│   ├── grid_search.py
│   ├── optuna_optimization.py
│   ├── preprocessing.py
│   ├── train_and_evaluate.py
│   └── xai_interpretation.py
```

---

## **Future Work**

* Multi-disorder EEG classification (PTSD vs MDD vs GAD)
* Cross-session and cross-device generalization
* Clinical tool: subject-level risk dashboard using SHAP/LIME profiles
* Multimodal fusion with **ECG**, **behavioral**, or **speech biomarkers**

---

## **Contributors**

**Lead Researchers**

* [ParmarDarshan29](https://github.com/ParmarDarshan29)
* [twi-exe](https://github.com/twi-exe)

**Supervision**

* Dr. Shankar Parmar
