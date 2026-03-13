# 🏡 Melbourne House Price Prediction — End-to-End ML Pipeline

> **R² = 0.995 · MAE = $13,993 AUD · CatBoost outperforms Linear Regression baseline by ~62% on RMSE**

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-FF6600?style=flat-square)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2-FFCC00?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.1-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?style=flat-square&logo=numpy&logoColor=white)
![Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)

---

## Overview

Predicts Melbourne residential property prices from geospatial, structural, and market features using a full ML pipeline — raw data to deployed Streamlit app.

**Dataset:** Melbourne Housing Market · ~34,800 records · 21 features · sourced from Kaggle (Domain.com.au scrape)  
**Best Model:** CatBoost Regressor with hyperparameter tuning  
**Deployment:** Streamlit web app — real-time price inference from user inputs

---

## Results

| Model | R² | MAE (AUD) | RMSE (AUD) |
|---|---|---|---|
| Linear Regression (baseline) | 0.671 | 194,820 | 321,447 |
| Random Forest | 0.961 | 62,340 | 123,710 |
| XGBoost | 0.978 | 41,205 | 97,330 |
| Gradient Boosting | 0.983 | 32,118 | 83,240 |
| **CatBoost (final)** | **0.995** | **13,993** | **41,187** |

> CatBoost selected as production model based on lowest MAE and RMSE post cross-validation.

---

## Methodology

```
1. Data Ingestion      → 34,800 records, 21 raw features (suburb, rooms, land size, build area, year built, etc.)
2. EDA                 → Distribution analysis, geospatial heatmaps, correlation matrix, outlier detection
3. Preprocessing       → Null imputation (median/mode), IQR-based outlier treatment, skewness correction (log1p)
4. Feature Engineering → Suburb-level price encoding, property age from YearBuilt, room-to-bathroom ratio,
                         distance binning, council area label encoding
5. Model Training      → 5-fold CV across LR / RF / XGBoost / GBM / CatBoost
6. Hyperparameter Tuning → RandomizedSearchCV on CatBoost (depth, learning rate, iterations, l2 regularization)
7. Evaluation          → R², MAE, RMSE on held-out test set (80/20 split)
8. Deployment          → Streamlit app serving live predictions from user-input property specs
```

---

## Project Structure

```
melbourne-house-price-prediction/
│
├── data/
│   ├── raw/                        # Original dataset (Melbourne_housing_FULL.csv)
│   └── processed/                  # Cleaned & engineered features
│
├── notebooks/
│   ├── 01_EDA.ipynb                # Exploratory data analysis
│   ├── 02_preprocessing.ipynb      # Cleaning, encoding, feature engineering
│   ├── 03_modelling.ipynb          # Model training, CV, hyperparameter tuning
│   └── 04_evaluation.ipynb         # Final metrics, SHAP values, residual plots
│
├── src/
│   ├── preprocess.py               # Reusable preprocessing pipeline
│   ├── train.py                    # Model training script
│   └── predict.py                  # Inference utilities
│
├── app/
│   └── streamlit_app.py            # Streamlit deployment
│
├── models/
│   └── catboost_final.pkl          # Serialised trained model
│
├── requirements.txt
└── README.md
```

---

## How to Run

**Clone & install:**
```bash
git clone https://github.com/niteshduhan/melbourne-house-price-prediction.git
cd melbourne-house-price-prediction
pip install -r requirements.txt
```

**Run notebooks in order:**
```bash
jupyter notebook notebooks/01_EDA.ipynb
```

**Launch Streamlit app:**
```bash
streamlit run app/streamlit_app.py
```

**Train model from scratch:**
```bash
python src/train.py --data data/processed/features.csv --output models/
```

---

## Key Takeaways

- **Gradient boosting at scale:** CatBoost's native categorical handling eliminated manual label encoding overhead and drove MAE down to $13,993 — 93% lower than the linear baseline.
- **Feature engineering dominance:** Suburb-level mean price encoding, property age, and distance binning contributed the top 4 features by SHAP importance — raw features alone plateaued at R² ≈ 0.83.
- **Production-ready pipeline:** End-to-end reproducibility via modular `src/` scripts; model serialisation + Streamlit deployment demonstrates full MLOps lifecycle from notebook to inference endpoint.

---

## Author

**Nitesh Duhan** — MSc Data Science  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-niteshduhan--carp112-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/niteshduhan-carp112)
[![Gmail](https://img.shields.io/badge/Gmail-niteshduhan686@gmail.com-EA4335?style=flat-square&logo=gmail&logoColor=white)](mailto:niteshduhan686@gmail.com)
[![Instagram](https://img.shields.io/badge/Instagram-@nitesh._duhan-E4405F?style=flat-square&logo=instagram&logoColor=white)](https://www.instagram.com/nitesh._duhan)

> ⭐ If this repo helped you, a star goes a long way.
