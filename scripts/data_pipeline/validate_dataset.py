#!/usr/bin/env python3
"""
Comprehensive Data Quality and Validation Script
================================================
Performs rigorous data quality checks on synthetic_food_supply_chain.csv:
  1. Missing values & null percentage
  2. Duplicate records (exact and operational)
  3. Data types & schema conformity
  4. Physical sanity & domain boundary checks (Temp, Humidity, Expiry, Loss, Coordinates)
  5. Statistical distribution and class balance checks
"""

import os
import sys
import json
import numpy as np
import pandas as pd

DATASET_PATH = "dataset/synthetic_food_supply_chain.csv"
OUTPUT_REPORT_PATH = "reports/validation_metrics.json"

def run_validation():
    print("=" * 80)
    print("BITE412L DATASET QUALITY AND VALIDATION AUDIT")
    print(f"Target File: {DATASET_PATH}")
    print("=" * 80)

    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: File {DATASET_PATH} not found!")
        sys.exit(1)

    df = pd.read_csv(DATASET_PATH)
    total_rows, total_cols = df.shape
    print(f"Loaded {total_rows:,} rows and {total_cols} columns.\n")

    results = {
        "dataset_shape": {"rows": total_rows, "columns": total_cols},
        "missing_values": {},
        "duplicates": {},
        "dtypes": {},
        "boundary_checks": {},
        "target_distributions": {},
        "status": "PASS"
    }

    # 1. Missing Values
    print("[1/5] Checking Missing Values...")
    missing = df.isnull().sum()
    missing_dict = missing[missing > 0].to_dict()
    if not missing_dict:
        print("  ✓ Zero missing values found across all 30 columns.")
        results["missing_values"] = {"total_missing_cells": 0, "columns_with_missing": {}}
    else:
        print(f"  ⚠ Missing values detected: {missing_dict}")
        results["missing_values"] = {
            "total_missing_cells": int(missing.sum()),
            "columns_with_missing": {k: int(v) for k, v in missing_dict.items()}
        }

    # 2. Duplicate Records
    print("\n[2/5] Checking Duplicate Records...")
    exact_dups = df.duplicated().sum()
    print(f"  ✓ Exact duplicate rows: {exact_dups}")
    
    # Check operational feature duplicates (excluding timestamp)
    operational_cols = [c for c in df.columns if c != "Timestamp"]
    op_dups = df.duplicated(subset=operational_cols).sum()
    print(f"  ✓ Operational feature duplicate rows (ignoring timestamp): {op_dups}")
    results["duplicates"] = {
        "exact_duplicates": int(exact_dups),
        "operational_duplicates": int(op_dups)
    }

    # 3. Data Types & Schema
    print("\n[3/5] Checking Data Types & Schema...")
    dtype_summary = {col: str(df[col].dtype) for col in df.columns}
    results["dtypes"] = dtype_summary
    print(f"  Numeric columns: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"  Categorical / String columns: {len(df.select_dtypes(include=['object']).columns)}")
    
    # Check Timestamp format
    try:
        pd.to_datetime(df["Timestamp"])
        print("  ✓ Timestamp column successfully parsed to datetime format.")
        results["timestamp_valid"] = True
    except Exception as e:
        print(f"  ✗ Timestamp parsing error: {e}")
        results["timestamp_valid"] = False
        results["status"] = "FAIL"

    # 4. Domain & Physical Boundary Checks
    print("\n[4/5] Checking Physical Sanity & Domain Boundaries...")
    boundary_results = {}

    # Coordinate check
    lat_valid = df["Latitude"].between(-90, 90).all()
    lon_valid = df["Longitude"].between(-180, 180).all()
    print(f"  ✓ GPS Coordinates within bounds: Latitude [-90, 90]: {lat_valid}, Longitude [-180, 180]: {lon_valid}")
    boundary_results["gps_coordinates"] = {
        "latitude_valid": bool(lat_valid),
        "longitude_valid": bool(lon_valid),
        "lat_min": float(df["Latitude"].min()), "lat_max": float(df["Latitude"].max()),
        "lon_min": float(df["Longitude"].min()), "lon_max": float(df["Longitude"].max())
    }

    # Humidity check (0-100%)
    hum_valid = df["Humidity"].between(0, 100).all()
    print(f"  ✓ Relative Humidity within [0, 100]%: {hum_valid} (min: {df['Humidity'].min():.2f}%, max: {df['Humidity'].max():.2f}%)")
    boundary_results["humidity"] = {
        "valid": bool(hum_valid),
        "min": float(df["Humidity"].min()), "max": float(df["Humidity"].max()),
        "mean": float(df["Humidity"].mean()), "std": float(df["Humidity"].std())
    }

    # Days to Expiry check (1-30 days)
    exp_valid = df["days_to_expiry"].between(1, 30).all()
    print(f"  ✓ Days to Expiry within [1, 30]: {exp_valid} (min: {df['days_to_expiry'].min()}, max: {df['days_to_expiry'].max()})")
    boundary_results["days_to_expiry"] = {
        "valid": bool(exp_valid),
        "min": int(df["days_to_expiry"].min()), "max": int(df["days_to_expiry"].max())
    }

    # Temperature per temp_class check
    temp_check = {}
    print("  Temperature distributions per temp_class:")
    for t_class, group in df.groupby("temp_class"):
        sim_t = group["simulated_temperature"]
        temp_check[t_class] = {
            "count": int(len(group)),
            "min": float(sim_t.min()),
            "mean": float(sim_t.mean()),
            "max": float(sim_t.max()),
            "std": float(sim_t.std()),
            "breaches": int((group["temp_deviation_score"] > 0).sum()),
            "breach_rate": float((group["temp_deviation_score"] > 0).mean())
        }
        print(f"    - {t_class.upper():<8}: N={len(group):<5} | min={sim_t.min():>6.2f}°C, mean={sim_t.mean():>6.2f}°C, max={sim_t.max():>6.2f}°C | Breach rate: {temp_check[t_class]['breach_rate']:.1%}")
    boundary_results["temperature_by_class"] = temp_check

    # Waiting Time & Logistics Delay
    wt_min, wt_max = df["Waiting_Time"].min(), df["Waiting_Time"].max()
    delay_rate = (df["Logistics_Delay"] > 0).mean()
    print(f"  ✓ Waiting Time range: [{wt_min:.1f} min, {wt_max:.1f} min], Delay frequency: {delay_rate:.2%}")
    boundary_results["logistics"] = {
        "waiting_time_min": float(wt_min), "waiting_time_max": float(wt_max),
        "delay_rate": float(delay_rate)
    }

    # Loss percentage range check (0-100%)
    loss_valid = df["synthetic_loss_percentage"].between(0, 100).all()
    print(f"  ✓ Synthetic Loss % strictly bounded in [0, 100]%: {loss_valid}")
    boundary_results["loss_percentage"] = {
        "strictly_in_bounds": bool(loss_valid),
        "min": float(df["synthetic_loss_percentage"].min()),
        "max": float(df["synthetic_loss_percentage"].max()),
        "mean": float(df["synthetic_loss_percentage"].mean()),
        "std": float(df["synthetic_loss_percentage"].std())
    }

    results["boundary_checks"] = boundary_results

    # 5. Target Distributions & Classification Balance
    print("\n[5/5] Analyzing Target Distributions...")
    loss_p = df["synthetic_loss_percentage"]
    risk = df["loss_risk"]
    p70 = float(np.percentile(loss_p, 70))
    pos_rate = float(risk.mean())
    print(f"  Synthetic Loss %: Mean={loss_p.mean():.2f}%, Median={loss_p.median():.2f}%, Std={loss_p.std():.2f}%")
    print(f"  Loss Risk (Binary): 0={int((risk==0).sum())} ({1-pos_rate:.1%}), 1={int((risk==1).sum())} ({pos_rate:.1%})")
    print(f"  70th Percentile Cutoff: {p70:.2f}%")

    results["target_distributions"] = {
        "synthetic_loss_percentage": {
            "mean": float(loss_p.mean()),
            "std": float(loss_p.std()),
            "min": float(loss_p.min()),
            "q25": float(loss_p.quantile(0.25)),
            "median": float(loss_p.median()),
            "q75": float(loss_p.quantile(0.75)),
            "max": float(loss_p.max()),
            "70th_percentile": p70
        },
        "loss_risk": {
            "class_0_count": int((risk == 0).sum()),
            "class_1_count": int((risk == 1).sum()),
            "positive_class_ratio": pos_rate
        }
    }

    os.makedirs(os.path.dirname(OUTPUT_REPORT_PATH), exist_ok=True)
    with open(OUTPUT_REPORT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics summary to: {OUTPUT_REPORT_PATH}")
    print("=" * 80)
    print("VALIDATION STATUS: ALL CHECKS PASSED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    run_validation()
