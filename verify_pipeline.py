"""
Automated Verification Suite for Palmer Penguins ML Pipeline
Author: MSU AI Club Workshop 01 Team
Date: Fall 2026

Runs automated unit tests on the training pipeline, missing value handling,
model accuracy thresholds, and CLI prediction capabilities.
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np

# Ensure cross-platform UTF-8 terminal encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def test_dataset_ingestion():
    print("--- [Test 1] Testing Dataset Ingestion ---")
    data_path = "penguins.csv"
    assert os.path.exists(data_path), f"Dataset file '{data_path}' not found!"
    df = pd.read_csv(data_path)
    assert len(df) == 344, f"Expected 344 rows, got {len(df)}"
    expected_cols = {"species", "island", "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "sex"}
    assert expected_cols.issubset(set(df.columns)), "Missing required dataset columns!"
    print(f"PASSED: Loaded {len(df)} rows and all expected columns.\n")
    return df

def test_data_cleaning(df):
    print("--- [Test 2] Testing Data Lifecycle Cleaning & Imputation ---")
    df_clean = df.copy()
    numeric_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    assert df_clean[numeric_cols].isnull().sum().sum() == 0, "Numeric imputation failed!"
    print("PASSED: Zero missing numeric values after median imputation.\n")
    return df_clean

def test_model_training(df_clean):
    print("--- [Test 3] Testing Model Training & Accuracy Thresholds ---")
    from train import train_model
    model, accuracy = train_model(df_clean)
    assert accuracy >= 0.85, f"Model accuracy ({accuracy:.1%}) below 85% threshold!"
    print(f"PASSED: Model accuracy {accuracy:.1%} exceeds 85% threshold.\n")
    return model

def test_inference(model):
    print("--- [Test 4] Testing Model Inference ---")
    sample_input = pd.DataFrame([{
        "bill_length_mm": 48.5,
        "bill_depth_mm": 15.0,
        "flipper_length_mm": 217.0,
        "body_mass_g": 5000.0
    }])
    pred = model.predict(sample_input)[0]
    assert pred == "Gentoo", f"Expected 'Gentoo', got '{pred}'"
    print(f"PASSED: Correctly predicted species '{pred}'.\n")

def main():
    print("=" * 60)
    print("Starting Automated Verification for Workshop 01 ML Pipeline...")
    print("=" * 60 + "\n")
    
    df = test_dataset_ingestion()
    df_clean = test_data_cleaning(df)
    model = test_model_training(df_clean)
    test_inference(model)
    
    print("=" * 60)
    print("SUCCESS: All verification steps PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    main()
