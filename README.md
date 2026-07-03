# BDPA-01: Predicting 30-Day Hospital Readmission in Diabetic Patients

Code and notebooks accompanying the technical report / preprint:

> Rajpoot, B., Rastogi, A., Sawant, R., Dige, T. and Deshmukh, D. (2026). *Predicting 30-Day Hospital Readmission in Diabetic Patients: A Comparative Evaluation of Class-Imbalance Correction, Feature Selection, and Clustering-Informed Classifiers.* [preprint link — add once posted]

This project investigates predicting 30-day hospital readmission using the [UCI Diabetes 130-US Hospitals dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) (Strack et al., 2014), comparing a conservative baseline pipeline against an improved pipeline that combines recursive feature elimination, k-means clustering, and SMOTE-based oversampling with hyperparameter tuning.

## Results summary

| Pipeline | Precision | Recall | F1-score | Accuracy |
|---|---|---|---|---|
| Baseline, no imbalance correction | 0.00 | 0.00 | 0.00 | 0.91 |
| Baseline + RandomOverSampler | 0.12 | 0.59 | 0.20 | 0.58 |
| Improved: RFE + k-means + SMOTE + tuning | 0.63 | 0.71 | 0.67 | 0.64 |

Metrics are for the positive (readmitted) class on the held-out test set. See the paper for full methodology, error analysis, and discussion.

## Repository structure

| File | Description |
|---|---|
| `DataCleaningPart1_Bhuvan_Considering.ipynb` | Initial data cleaning and preprocessing for the baseline pipeline (missing-value handling, outlier removal, feature encoding). Corresponds to paper Section 3.2 and Section 4.1. |
| `DataVisualization_Tejas_Considering.ipynb` | Exploratory data analysis and plots (class distribution, age/gender breakdowns, correlation matrix). Corresponds to paper Section 3.3 and Figures 1–2. |
| `ModelBuilding_Devshree_Considering.ipynb` | Baseline logistic regression model, cross-validation, and RandomOverSampler experiments. Corresponds to paper Section 4.2–4.3 and Tables 1–2. |
| `Improved_Model_Data_cleaning_Tejas_Considerd.ipynb` | Revised, retention-prioritised preprocessing for the improved pipeline. Corresponds to paper Section 5.1. |
| `Improved_Model_Devshree_considered.ipynb` | Recursive feature elimination, k-means clustering, SMOTE, and GridSearchCV hyperparameter tuning. Corresponds to paper Sections 5.2–5.4 and Tables 3–4, Figure 3. |
| `Report.pdf` | Original University of Leicester coursework report (CO7093 Big Data & Predictive Analytics) this project and the paper are based on. |
| `Requirements.txt` | Python package versions used. |

> **Note:** Notebook filenames retain their original working-draft names from the group coursework submission (e.g. `_Considering`, `_Considerd`) rather than being renamed retrospectively, so that they remain traceable to the original coursework history.

## Setup

```bash
git clone https://github.com/bhuvan-01/BDPA-01.git
cd BDPA-01
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r Requirements.txt
```

Requires Python 3.12. Key dependencies: `pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`, `plotly`, `imblearn` (imbalanced-learn) — see `Requirements.txt` for pinned versions.

Download the dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) and place `diabetic_data.csv` in the repository root (or update the file path at the top of each notebook) before running.

## Running the notebooks

Notebooks are intended to be run in the following order to reproduce the paper's results:

1. `DataCleaningPart1_Bhuvan_Considering.ipynb` → `DataVisualization_Tejas_Considering.ipynb` → `ModelBuilding_Devshree_Considering.ipynb` (baseline pipeline, Tables 1–2)
2. `Improved_Model_Data_cleaning_Tejas_Considerd.ipynb` → `Improved_Model_Devshree_considered.ipynb` (improved pipeline, Tables 3–4)

## Data source

Strack, B., DeShazo, J.P., Gennings, C., Olmo, J.L., Ventura, S., Cios, K.J. and Clore, J.N. (2014) 'Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records', *BioMed Research International*, 2014, Article ID 781670. https://doi.org/10.1155/2014/781670

## Contributors

Bhuvan Rajpoot, Amisha Rastogi, Rahul Sawant, Tejas Dige, Devershree Deshmukh — School of Computing and Mathematical Sciences, University of Leicester.

## License

[MIT](LICENSE) — add a `LICENSE` file if you'd like this explicitly stated; otherwise the code defaults to standard copyright with all rights reserved.

## Citation

If you reference this work, please cite the accompanying preprint (see top of this file) rather than this repository directly.
