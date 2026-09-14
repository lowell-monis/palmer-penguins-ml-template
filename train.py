"""
Palmer Penguins Species Classifier & Data Lifecycle Training Pipeline
Template Repository for MSU AI Club Workshop 01

DATA DOWNLOAD INSTRUCTIONS:
This script automatically downloads the Palmer Penguins dataset from the public URL below.
Never commit raw dataset files directly to git repositories!

Public Data URL:
    https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Ensure cross-platform UTF-8 terminal encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Public Dataset URL
DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"

def load_data(url=DATA_URL):
    """
    STAGE 1: DATA INGESTION VIA PUBLIC URL
    --------------------------------------
    Fetch dataset dynamically from remote URL.
    Never commit raw data CSV files directly to git repositories!
    """
    print(f"[Stage 1: Ingestion] Downloading dataset from public URL: {url}")
    df = pd.read_csv(url)
    print(f"                     Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"                     Missing values count by column:")
    for col, null_count in df.isnull().sum().items():
        print(f"                       * {col:20s}: {null_count} missing values")
    return df

def clean_data(df):
    """
    STAGE 2: DATA PROCESSING & IMPUTATION
    -------------------------------------
    Perform median imputation for continuous features and mode imputation for categorical attributes.
    """
    df_clean = df.copy()
    
    numeric_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            print(f"                     Imputed missing '{col}' with median: {median_val:.1f}")
            
    if df_clean["sex"].isnull().sum() > 0:
        mode_sex = df_clean["sex"].mode()[0]
        df_clean["sex"] = df_clean["sex"].fillna(mode_sex)
        print(f"                     Imputed missing 'sex' with mode: {mode_sex}")
        
    return df_clean

def train_model(df):
    """
    STAGE 3 & 4: FEATURE ENGINEERING, TRAINING & COHORT EVALUATION
    --------------------------------------------------------------
    Evaluate per-class precision and recall using stratified train/test split.
    """
    feature_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    X = df[feature_cols]
    y = df["species"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "=" * 55)
    print(f"[Stage 4: Evaluation] Overall Model Accuracy: {accuracy:.1%}")
    print("=" * 55)
    print("\n--- Per-Species Classification Report ---")
    print(classification_report(y_test, y_pred))
    
    print("--- Feature Importance Ranks ---")
    for feat, imp in zip(feature_cols, clf.feature_importances_):
        print(f"  * {feat:20s}: {imp:.4f}")
        
    return clf, accuracy

def export_model(model, output_path="penguin_model.pkl"):
    """
    STAGE 5: ARTIFACT DEPLOYMENT & EXPORT
    -------------------------------------
    Serialize trained model to file for real-time CLI and web app predictions.
    """
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\n[Stage 5: Export] Serialized model saved to: {output_path}")

def main():
    print("Starting Palmer Penguins ML Classifier & Data Lifecycle Training...\n")
    df_raw = load_data()
    df_clean = clean_data(df_raw)
    model, accuracy = train_model(df_clean)
    export_model(model)
    print("\n[Success] Training Pipeline Completed Successfully!")

if __name__ == "__main__":
    main()
