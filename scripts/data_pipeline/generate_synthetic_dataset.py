"""
Synthetic Food Supply Chain Dataset Generator
================================================
Fuses three real data sources into one synthetic, sensor-shaped dataset
suitable for training the loss/quality prediction model in the
BITE412L Cloud Intelligence Platform project.

SOURCES (download these yourself, links below):
  1. FAO Food Loss and Waste (FLW) Database
     https://www.fao.org/platform-food-loss-waste/flw-data/
     -> gives REAL loss_percentage distributions per commodity/stage/cause
  2. Smart Logistics Supply Chain Dataset (Kaggle, ziya07)
     https://www.kaggle.com/datasets/ziya07/smart-logistics-supply-chain-dataset
     -> gives REAL-shaped IoT sensor stream (temperature, humidity, delay, etc.)
  3. Retail Food Package Expiry Date Dataset (Kaggle, ziya07)
     https://www.kaggle.com/datasets/ziya07/retail-food-package-expiry-date-dataset
     -> gives REAL-shaped shelf-life / days-to-expiry data

HOW IT WORKS
------------
- The Smart Logistics rows are the backbone (one row = one sensor reading
  at one point in the simulated supply chain).
- Each row is assigned a commodity + food_supply_stage sampled from the
  REAL FAO commodity/stage frequency distribution.
- FAO's real (commodity, stage) loss_percentage becomes the BASE RATE.
- Commodities are classified into temp classes ('frozen', 'cold', 'ambient')
  and temperatures are rescaled into realistic bands with occasional breach injection.
- Farm/Harvest stages are realistically remapped to 'Storage (post-harvest collection)'
  for display while keeping raw 'fao_stage' for accurate lookup.
- The base rate is then perturbed using the row's sensor values
  (temperature deviation, humidity deviation, logistics delay, and
  days-to-expiry pressure from the retail dataset) to produce a final
  synthetic loss_percentage / loss_risk label.
- Optional --target-rows expands the logistics backbone via bootstrap
  resampling + Gaussian jitter to match FAO's scale (e.g. 30,000 rows).
"""

import argparse
import os
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# ----------------------------------------------------------------------
# CONFIG -- matched to your downloaded CSV headers
# ----------------------------------------------------------------------

FAO_COLS = {
    "commodity": "commodity",
    "stage": "food_supply_stage",
    "cause": "cause_of_loss",
    "loss_pct": "loss_percentage",
    "sample_size": "sample_size",
}

LOGISTICS_COLS = {
    "timestamp": "Timestamp",
    "asset_id": "Asset_ID",
    "temperature": "Temperature",
    "humidity": "Humidity",
    "inventory_level": "Inventory_Level",
    "shipment_status": "Shipment_Status",
    "waiting_time": "Waiting_Time",
    "delay_flag": "Logistics_Delay",
    "delay_reason": "Logistics_Delay_Reason",
    "asset_utilization": "Asset_Utilization",
    "demand_forecast": "Demand_Forecast",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "traffic_status": "Traffic_Status",
    "user_transaction_amount": "User_Transaction_Amount",
    "user_purchase_frequency": "User_Purchase_Frequency",
}

EXPIRY_COLS = {
    "expiry_text": "expiry_text",
    "surface_type": "surface_type",
    "image_name": "image_name",
    "category": "Category",
    "days_to_expiry": "days_to_expiry",
}

# "Whole supply chain" describes the entire chain, not one point in it --
# dropped from the sampling pool entirely.
EXCLUDE_STAGES = {"whole supply chain"}

# "Farm" and "Harvest" are real FAO stages with loss statistics.
# A truck row reading can plausibly represent the post-harvest collection/storage point.
STAGE_DISPLAY_REMAP = {
    "farm": "Storage (post-harvest collection)",
    "harvest": "Storage (post-harvest collection)",
}

# Commodities are classified into a rough cold-chain requirement class by
# keyword match. This drives which temperature band counts as "safe".
FROZEN_KEYWORDS = ["frozen", "ice cream"]
COLD_KEYWORDS = ["dairy", "milk", "yogurt", "yoghurt", "cheese", "meat",
                  "egg", "fish", "poultry", "seafood", "butter"]

TEMP_BAND_BY_CLASS = {
    "frozen": (-18, -12),
    "cold": (0, 8),
    "ambient": (10, 25),
}

STAGE_HUMIDITY_BAND = {
    "default": (40, 70),
}


def classify_commodity_temp_class(commodity):
    c = str(commodity).lower()
    if any(k in c for k in FROZEN_KEYWORDS):
        return "frozen"
    if any(k in c for k in COLD_KEYWORDS):
        return "cold"
    return "ambient"


def to_display_stage(fao_stage):
    """Real FAO stage -> label attached to the output row. Only Farm/Harvest
    get remapped; everything else passes through unchanged."""
    return STAGE_DISPLAY_REMAP.get(str(fao_stage).lower(), fao_stage)


# ----------------------------------------------------------------------
# LOADERS (with mock fallback so the script always runs end-to-end)
# ----------------------------------------------------------------------

def load_fao(path):
    if path and os.path.exists(path):
        df = pd.read_csv(path, low_memory=False)
        if FAO_COLS["loss_pct"] in df.columns:
            df[FAO_COLS["loss_pct"]] = pd.to_numeric(df[FAO_COLS["loss_pct"]], errors="coerce")
            df = df.dropna(subset=[FAO_COLS["commodity"], FAO_COLS["stage"], FAO_COLS["loss_pct"]])
        return df
    print("[warn] FAO file not found -- using MOCK FAO data.")
    commodities = ["Dairy products", "Fruits", "Vegetables", "Cereals", "Meat"]
    stages = ["Storage", "Transport", "Retail"]
    causes = ["Poor temperature control", "Mechanical damage", "Pest infestation", "Delays"]
    rows = []
    for c in commodities:
        for s in stages:
            for cause in causes:
                rows.append({
                    FAO_COLS["commodity"]: c,
                    FAO_COLS["stage"]: s,
                    FAO_COLS["cause"]: cause,
                    FAO_COLS["loss_pct"]: max(0, RNG.normal(10, 5)),
                    FAO_COLS["sample_size"]: RNG.integers(20, 400),
                })
    return pd.DataFrame(rows)


def load_logistics(path, n_mock=2000):
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        return df
    print("[warn] Logistics file not found -- using MOCK logistics data.")
    timestamps = pd.date_range("2024-01-01", periods=n_mock, freq="h")
    df = pd.DataFrame({
        LOGISTICS_COLS["timestamp"]: timestamps,
        LOGISTICS_COLS["asset_id"]: RNG.integers(1000, 1050, n_mock),
        LOGISTICS_COLS["temperature"]: RNG.normal(6, 5, n_mock),   # deg C
        LOGISTICS_COLS["humidity"]: RNG.normal(55, 15, n_mock),    # %
        LOGISTICS_COLS["inventory_level"]: RNG.integers(0, 500, n_mock),
        LOGISTICS_COLS["shipment_status"]: RNG.choice(
            ["In Transit", "Delivered", "Delayed"], n_mock),
        LOGISTICS_COLS["waiting_time"]: RNG.exponential(20, n_mock),
        LOGISTICS_COLS["delay_flag"]: RNG.choice([0, 1], n_mock, p=[0.8, 0.2]),
        LOGISTICS_COLS["delay_reason"]: RNG.choice(
            ["None", "Weather", "Mechanical Failure", "Traffic"], n_mock),
        LOGISTICS_COLS["asset_utilization"]: RNG.uniform(30, 95, n_mock),
        LOGISTICS_COLS["demand_forecast"]: RNG.integers(50, 400, n_mock),
    })
    return df


def load_expiry(path, n_mock=2000):
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        # If expiry_text is present, calculate remaining days to expiry
        if EXPIRY_COLS["expiry_text"] in df.columns:
            parsed_dates = pd.to_datetime(df[EXPIRY_COLS["expiry_text"]], format="mixed", dayfirst=True, errors="coerce")
            min_date = parsed_dates.min()
            df["days_to_expiry"] = ((parsed_dates - min_date).dt.days % 30 + 1).fillna(15).astype(int)
        elif "days_to_expiry" not in df.columns and EXPIRY_COLS["days_to_expiry"] not in df.columns:
            df["days_to_expiry"] = RNG.integers(1, 30, len(df))
        return df
    print("[warn] Expiry file not found -- using MOCK expiry data.")
    df = pd.DataFrame({
        EXPIRY_COLS["category"]: RNG.choice(
            ["Dairy products", "Fruits", "Vegetables", "Cereals", "Meat"], n_mock),
        "days_to_expiry": RNG.integers(1, 30, n_mock),
        "surface_type": RNG.choice(["plastic", "cardboard", "foil", "paper", "glass"], n_mock),
    })
    return df


def expand_with_jitter(df, target_n, time_col=None, jitter_frac=0.03):
    """Bootstrap-resample a smaller real dataset up to target_n rows.

    Every value still originates from a real observation -- this is
    resampling-with-replacement plus light noise, not fabrication from
    nothing. Used when a real source (logistics, expiry) is smaller
    than the FAO backbone and you want the output volume to match FAO's
    scale.

    - If target_n <= len(df): returns a random subsample.
    - If target_n > len(df): resamples with replacement, then:
        * adds Gaussian noise (std * jitter_frac) to numeric columns so
          repeated draws of the same original row aren't byte-identical
        * if time_col is given, spreads timestamps across a proportionally
          wider window instead of leaving thousands of rows stacked on
          the same handful of original timestamps
    """
    n = len(df)
    if target_n <= n:
        return df.sample(n=target_n, random_state=42).reset_index(drop=True)

    idx = RNG.integers(0, n, target_n)
    expanded = df.iloc[idx].reset_index(drop=True)

    numeric_cols = expanded.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        std = expanded[col].std()
        if pd.notna(std) and std > 0:
            noise = RNG.normal(0, std * jitter_frac, len(expanded))
            expanded[col] = expanded[col] + noise
            if (df[col] >= 0).all():
                expanded[col] = expanded[col].clip(lower=0)

    if time_col and time_col in expanded.columns:
        start = pd.to_datetime(df[time_col]).min()
        end = pd.to_datetime(df[time_col]).max()
        span_seconds = max((end - start).total_seconds(), 3600)
        extended_seconds = span_seconds * (target_n / n)
        offsets = RNG.uniform(0, extended_seconds, target_n)
        expanded[time_col] = pd.to_datetime(start) + pd.to_timedelta(offsets, unit="s")
        expanded = expanded.sort_values(time_col).reset_index(drop=True)

    return expanded


# ----------------------------------------------------------------------
# FUSION LOGIC
# ----------------------------------------------------------------------

def build_fao_lookup(fao_df):
    """Real (commodity, stage) -> (mean, std) loss_percentage from FAO."""
    g = fao_df.groupby([FAO_COLS["commodity"], FAO_COLS["stage"]])[FAO_COLS["loss_pct"]]
    stats = g.agg(["mean", "std", "count"]).reset_index()
    stats["std"] = stats["std"].fillna(stats["mean"] * 0.3)
    overall_mean = fao_df[FAO_COLS["loss_pct"]].mean()
    overall_std = fao_df[FAO_COLS["loss_pct"]].std()
    return stats, overall_mean, overall_std


def sample_commodity_stage(fao_df, n):
    """Sample commodity/stage pairs weighted by their real FAO frequency.

    Rows whose stage is in EXCLUDE_STAGES (e.g. 'Whole supply chain') are
    dropped first so they correspond to a specific stage in the pipeline.
    """
    pairs = fao_df[[FAO_COLS["commodity"], FAO_COLS["stage"]]].dropna()
    mask = ~pairs[FAO_COLS["stage"]].astype(str).str.lower().isin(EXCLUDE_STAGES)
    pairs = pairs[mask]
    if len(pairs) == 0:
        raise ValueError("No FAO rows left after excluding vague stages -- check EXCLUDE_STAGES.")
    idx = RNG.integers(0, len(pairs), n)
    sampled = pairs.iloc[idx].reset_index(drop=True)
    return sampled[FAO_COLS["commodity"]], sampled[FAO_COLS["stage"]]


def rescale_temperature(raw_temp, raw_min, raw_max, band, breach_prob=0.12):
    """Map the source dataset's raw temperature into a commodity-appropriate band,
    preserving its relative position, then occasionally inject a genuine cold-chain breach.
    """
    lo, hi = band
    if raw_max - raw_min < 1e-6:
        scaled = (lo + hi) / 2
    else:
        pct = (raw_temp - raw_min) / (raw_max - raw_min)
        scaled = lo + pct * (hi - lo)
    if RNG.random() < breach_prob:
        direction = RNG.choice([-1, 1])
        scaled += direction * RNG.uniform(3, 10)
    return scaled


def temp_deviation_score(temp, band):
    lo, hi = band
    if temp < lo:
        return (lo - temp) / max(1, (hi - lo))
    if temp > hi:
        return (temp - hi) / max(1, (hi - lo))
    return 0.0


def humidity_deviation_score(hum, stage):
    lo, hi = STAGE_HUMIDITY_BAND["default"]
    if hum < lo:
        return (lo - hum) / (hi - lo)
    if hum > hi:
        return (hum - hi) / (hi - lo)
    return 0.0


def expiry_pressure_score(days_left, max_days=30):
    days_left = max(0, min(days_left, max_days))
    return 1 - (days_left / max_days)   # closer to expiry -> closer to 1


def fuse(fao_df, logistics_df, expiry_df,
         w_temp=6.0, w_hum=3.0, w_delay=5.0, w_expiry=8.0, noise_std=2.0):
    n = len(logistics_df)
    stats, overall_mean, overall_std = build_fao_lookup(fao_df)
    stats_lookup = {
        (r[FAO_COLS["commodity"]], r[FAO_COLS["stage"]]): (r["mean"], r["std"])
        for _, r in stats.iterrows()
    }

    commodities, stages = sample_commodity_stage(fao_df, n)

    # attach expiry info and packaging material by sampling from the expiry dataset
    has_category = EXPIRY_COLS["category"] in expiry_df.columns
    expiry_col = "days_to_expiry" if "days_to_expiry" in expiry_df.columns else EXPIRY_COLS["days_to_expiry"]

    if has_category and expiry_col in expiry_df.columns:
        expiry_by_cat = {
            cat: sub[expiry_col].values
            for cat, sub in expiry_df.groupby(EXPIRY_COLS["category"])
        }
    else:
        expiry_by_cat = {}

    days_to_expiry = []
    packaging_materials = []
    expiry_pool = expiry_df[expiry_col].values if expiry_col in expiry_df.columns else None
    surface_pool = expiry_df[EXPIRY_COLS["surface_type"]].values if EXPIRY_COLS["surface_type"] in expiry_df.columns else None

    for c in commodities:
        pool = expiry_by_cat.get(c)
        if pool is not None and len(pool) > 0:
            days_to_expiry.append(RNG.choice(pool))
        elif expiry_pool is not None and len(expiry_pool) > 0:
            days_to_expiry.append(RNG.choice(expiry_pool))
        else:
            days_to_expiry.append(RNG.integers(1, 30))

        if surface_pool is not None and len(surface_pool) > 0:
            packaging_materials.append(RNG.choice(surface_pool))
        else:
            packaging_materials.append("standard")

    out = logistics_df.copy().reset_index(drop=True)
    out["commodity"] = commodities.values
    out["fao_stage"] = stages.values
    out["food_supply_stage"] = [to_display_stage(s) for s in stages.values]
    out["packaging_material"] = packaging_materials
    out["days_to_expiry"] = days_to_expiry

    # fill missing delay reasons if blank
    if LOGISTICS_COLS["delay_reason"] in out.columns:
        out[LOGISTICS_COLS["delay_reason"]] = out[LOGISTICS_COLS["delay_reason"]].fillna("Unknown")

    raw_temp_col = out.get(LOGISTICS_COLS["temperature"])
    raw_min, raw_max = (raw_temp_col.min(), raw_temp_col.max()) if raw_temp_col is not None else (0, 1)

    base_means, base_stds = [], []
    td_scores, hd_scores, ep_scores, loss_pcts = [], [], [], []
    temp_classes, raw_temps, sim_temps = [], [], []

    for i, row in out.iterrows():
        key = (row["commodity"], row["fao_stage"])
        mean, std = stats_lookup.get(key, (overall_mean, overall_std))
        base_means.append(mean)
        base_stds.append(std)

        raw_temp = row.get(LOGISTICS_COLS["temperature"], np.nan)
        hum = row.get(LOGISTICS_COLS["humidity"], np.nan)
        delay = row.get(LOGISTICS_COLS["delay_flag"], 0)

        temp_class = classify_commodity_temp_class(row["commodity"])
        band = TEMP_BAND_BY_CLASS[temp_class]
        sim_temp = (rescale_temperature(raw_temp, raw_min, raw_max, band)
                    if pd.notna(raw_temp) else (band[0] + band[1]) / 2)

        td = temp_deviation_score(sim_temp, band)
        hd = humidity_deviation_score(hum, row["fao_stage"]) if pd.notna(hum) else 0.0
        ep = expiry_pressure_score(row["days_to_expiry"])

        temp_classes.append(temp_class)
        raw_temps.append(raw_temp)
        sim_temps.append(sim_temp)
        td_scores.append(td)
        hd_scores.append(hd)
        ep_scores.append(ep)

        base = RNG.normal(mean, max(std, 0.5))
        adjustment = w_temp * td + w_hum * hd + w_delay * float(delay) + w_expiry * ep
        noise = RNG.normal(0, noise_std)
        loss_pct = np.clip(base + adjustment + noise, 0, 100)
        loss_pcts.append(loss_pct)

    out["temp_class"] = temp_classes
    out["raw_temperature"] = raw_temps
    out["simulated_temperature"] = sim_temps
    out["fao_base_loss_mean"] = base_means
    out["temp_deviation_score"] = td_scores
    out["humidity_deviation_score"] = hd_scores
    out["expiry_pressure_score"] = ep_scores
    out["synthetic_loss_percentage"] = loss_pcts

    # binary label for classification tasks -- threshold at the 70th percentile
    threshold = np.percentile(out["synthetic_loss_percentage"], 70)
    out["loss_risk"] = (out["synthetic_loss_percentage"] >= threshold).astype(int)

    return out


# ----------------------------------------------------------------------
# VALIDATION
# ----------------------------------------------------------------------

def validate(fao_df, synthetic_df):
    print("\n--- Validation summary ---")
    print(f"FAO real loss_percentage:      mean={fao_df[FAO_COLS['loss_pct']].mean():.2f}, "
          f"std={fao_df[FAO_COLS['loss_pct']].std():.2f}")
    print(f"Synthetic loss_percentage:     mean={synthetic_df['synthetic_loss_percentage'].mean():.2f}, "
          f"std={synthetic_df['synthetic_loss_percentage'].std():.2f}")
    print(f"loss_risk positive rate:       {synthetic_df['loss_risk'].mean():.2%}")
    print(f"Temperature deviation score:   min={synthetic_df['temp_deviation_score'].min():.2f}, "
          f"mean={synthetic_df['temp_deviation_score'].mean():.2f}, "
          f"max={synthetic_df['temp_deviation_score'].max():.2f}, "
          f"zero_rate={(synthetic_df['temp_deviation_score'] == 0).mean():.2%}")
    try:
        from scipy.stats import ks_2samp
        stat, p = ks_2samp(fao_df[FAO_COLS["loss_pct"]].dropna(),
                            synthetic_df["synthetic_loss_percentage"])
        print(f"KS test vs FAO distribution:   statistic={stat:.3f}, p-value={p:.4f}")
        print("(Very low p-value just means the two distributions differ in shape --")
        print(" expected, since yours is perturbed by sensor conditions. Use this as")
        print(" a sanity check, not a pass/fail gate.)")
    except ImportError:
        print("[info] scipy not installed -- skipping KS test.")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fao", default=None, help="Path to FAO FLW CSV")
    parser.add_argument("--logistics", default=None, help="Path to Smart Logistics CSV")
    parser.add_argument("--expiry", default=None, help="Path to Retail Expiry CSV")
    parser.add_argument("--out", default="synthetic_food_supply_chain.csv")
    parser.add_argument("--target-rows", type=int, default=30000,
                        help="Expand the logistics backbone to this many rows via "
                             "bootstrap resampling + jitter (default: 30000 to match FAO's scale).")
    args = parser.parse_args()

    fao_df = load_fao(args.fao)
    logistics_df = load_logistics(args.logistics)
    expiry_df = load_expiry(args.expiry)

    if args.target_rows:
        original_n = len(logistics_df)
        logistics_df = expand_with_jitter(
            logistics_df, args.target_rows, time_col=LOGISTICS_COLS["timestamp"])
        print(f"[info] Expanded logistics backbone from {original_n} real rows "
              f"to {len(logistics_df)} rows via bootstrap resampling + jitter.")

    synthetic_df = fuse(fao_df, logistics_df, expiry_df)
    synthetic_df.to_csv(args.out, index=False)

    validate(fao_df, synthetic_df)
    print(f"\nSaved {len(synthetic_df)} rows to {args.out}")
    print(synthetic_df[["commodity", "fao_stage", "food_supply_stage", "temp_class", "synthetic_loss_percentage", "loss_risk"]].head())


if __name__ == "__main__":
    main()
