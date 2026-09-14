# 🐧 Palmer Penguins Species Classifier & Data Lifecycle Pipeline
### *MSU AI Club Workshop 01 Template Repository*

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/Pipeline-Passing-brightgreen.svg)](verify_pipeline.py)

An end-to-end Machine Learning classification pipeline predicting Palmer Archipelago penguin species (`Adelie`, `Chinstrap`, `Gentoo`) based on biological measurements. Built with `pandas`, `scikit-learn`, `plotly`, and containerized CLI inference tools.

---

## 📊 Project Overview & Data Lifecycle Hints

This template repository executes a 5-stage Data Lifecycle pipeline on the Palmer Station Antarctica LTER dataset (`penguins.csv`):

1. **Ingestion**: Raw measurement collection across 344 penguin observations.  
   * *[Lifecycle Hint]*: Always inspect column data types, missingness frequencies, and unmeasured human/biological dimensions before training.
2. **Cleaning & Imputation**: Handling missing physical measurements with median/mode imputation.  
   * *[Lifecycle Hint]*: Avoid naive row deletion (`df.dropna()`) to prevent dropping underrepresented observation cohorts.
3. **Preprocessing**: Feature matrix formulation ($X$) and stratified train/test splitting.  
   * *[Lifecycle Hint]*: Stratified splitting ensures small classes (Chinstrap) maintain equal representation in train and test sets.
4. **Model Training & Evaluation**: Random Forest classification achieving **>95% accuracy**.  
   * *[Lifecycle Hint]*: Evaluate per-class precision and recall rather than relying solely on overall accuracy.
5. **CLI Inference**: Interactive command-line tool (`predict.py`) for real-time species predictions.  
   * *[Lifecycle Hint]*: Validate input schemas before passing measurements to serialized model artifacts (`penguin_model.pkl`).

---

## 💻 Running Options: Notebook vs. Modular Scripts

### Option A: Notebook Alternative (Live Demo & Visual Exploration)
Use `penguin_classifier.ipynb` for quick visual exploration, Plotly scatter charts, and live interactive execution:
```bash
jupyter notebook penguin_classifier.ipynb
```

### Option B: Modular Python Scripts (Production Repo Workflow)
```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn plotly jupyter

# 2. Train the classifier model (creates penguin_model.pkl)
python train.py

# 3. Predict species for custom penguin measurements via CLI
python predict.py --bill_length 48.5 --bill_depth 15.0 --flipper_length 217 --body_mass 5000

# 4. Run automated unit tests
python verify_pipeline.py
```

---

## 📁 Repository File Organization Guide

```
palmer-penguins-ml-template/
├── README.md                   # Project documentation & Data Lifecycle Model Card
├── penguins.csv                # Palmer Archipelago raw dataset (344 rows)
├── train.py                    # 5-stage Data Lifecycle training script
├── predict.py                  # CLI inference script
├── penguin_classifier.ipynb    # Notebook alternative (EDA & Plotly charts)
├── verify_pipeline.py          # Automated unit test suite (>85% accuracy check)
└── .gitignore                  # Git ignore rules for bytecode & model binaries
```

---

## 🎯 Model Performance & Metrics

| Species | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Adelie** | 97% | 97% | 97% | 30 |
| **Chinstrap** | 92% | 86% | 89% | 14 |
| **Gentoo** | 96% | 100% | 98% | 25 |
| **Overall** | **96%** | **96%** | **96%** | **69** |

---

## 📚 References

[1] K. B. Gorman, T. D. Williams, and W. R. Fraser, "Ecological sexual dimorphism and environmental variability within a community of Antarctic penguins (genus *Pygoscelis*)," *PLoS ONE*, vol. 9, no. 3, p. e90081, 2014.

[2] C. O'Neil, *Weapons of Math Destruction: How Big Data Increases Inequality and Threatens Democracy*. New York, NY, USA: Crown Publishing Group, 2016.

[3] C. D'Ignazio and L. F. Klein, *Data Feminism*. Cambridge, MA, USA: MIT Press, 2020.

[4] E. Yudkowsky and N. Soares, *If Anyone Builds It, Everyone Dies: Why Superhuman AI Would Kill Us All*. New York, NY, USA: Little, Brown and Company, 2025.

---

## 💙 Credits & License
Maintained by [**Lowell Monis**](https://lowell-monis.github.io/) & the **MSU AI Club Workshop Team**.  
© 2026 Michigan State University AI Club. All rights reserved. Released under the MIT License.
