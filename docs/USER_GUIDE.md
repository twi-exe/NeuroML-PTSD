# User Guide

This guide provides step-by-step instructions for setting up and using the NeuroML-PTSD project.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

- Python 3.8 or higher
- Git
- pip (Python package manager)

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/twi-exe/NeuroML-PTSD.git
   cd NeuroML-PTSD
   ```

2. **Set Up a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. **Data Preprocessing**
   - Place your EEG dataset in the `data/` directory.
   - Run the preprocessing script:
     ```bash
     python src/preprocessing.py
     ```

### 2. **Feature Selection**
   - Perform feature selection using ElasticNet:
     ```bash
     python pipeline/feature_selection.py
     ```

### 3. **Model Training**
   - Train machine learning models:
     ```bash
     python pipeline/train_and_evaluate.py
     ```

### 4. **Explainability Analysis**
   - Generate SHAP and LIME explanations:
     ```bash
     python pipeline/xai.py
     ```

### 5. **Evaluation**
   - Evaluate model performance:
     ```bash
     python pipeline/evaluation.py
     ```

## Jupyter Notebooks

For interactive analysis, explore the notebooks in the `notebooks/` directory:

- `prelim.ipynb`: Preliminary data exploration
- `visuals.ipynb`: Visualization of results
- `xai.ipynb`: Explainability analysis

## Results

Model outputs, SHAP plots, and other results are saved in the `results/` directory.

## Troubleshooting

If you encounter any issues, please check the following:

- Ensure all dependencies are installed.
- Verify that your dataset is correctly formatted.
- Refer to the [CONTRIBUTING.md](../CONTRIBUTING.md) file for reporting issues.