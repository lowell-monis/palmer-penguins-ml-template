# 🐧 Palmer Penguins Species Classifier & Data Lifecycle Pipeline
### *MSU AI Club Workshop 01 Template Repository*  
**Event Link**: [MSU AI Club Event Check-in Page](https://www.msuaiclub.com/events/df4d51d5-195a-4a8a-bd5e-db9c84133f81)  

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning classification pipeline predicting Palmer Archipelago penguin species (`Adelie`, `Chinstrap`, `Gentoo`) based on biological measurements. Built with `pandas`, `scikit-learn`, `plotly`, and CLI inference tools.

---

## 📊 Dataset Download & Ingestion

> **Best Practice Notice**: Raw data CSV files should **never** be committed directly to git repositories.

This project fetches the Palmer Archipelago dataset dynamically at runtime from the public Data URL:
```python
DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
```
Or download the dataset locally via terminal:
```bash
curl -O https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv
```

---

## 💻 Running Options: Notebook vs. Modular Scripts

### Option A: Notebook Alternative (Visual Exploration & Live Demo)
Run `penguin_classifier.ipynb` for step-by-step exploratory data analysis, Plotly scatter charts, and interactive pipeline execution:
```bash
jupyter notebook penguin_classifier.ipynb
```

### Option B: Modular Python Scripts (Production Repo Workflow)
```bash
# 1. Install required Python packages
pip install pandas numpy scikit-learn plotly jupyter

# 2. Train the Random Forest classifier model (creates penguin_model.pkl)
python train.py

# 3. Predict species for custom penguin bill/flipper measurements via CLI
python predict.py --bill_length 48.5 --bill_depth 15.0 --flipper_length 217 --body_mass 5000
```

---

## 🎯 Model Performance & Metrics

| Species | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Adelie** | 100% | 97% | 98% | 30 |
| **Chinstrap** | 93% | 100% | 97% | 14 |
| **Gentoo** | 100% | 100% | 100% | 25 |
| **Overall** | **98%** | **99%** | **98%** | **69** |

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
