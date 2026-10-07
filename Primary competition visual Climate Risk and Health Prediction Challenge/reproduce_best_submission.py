from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "Train.csv"
TEST_PATH = ROOT / "Test.csv"
CLIMATE_PATH = ROOT / "climate_features.csv"
OUTPUT_PATH = ROOT / "best_submission_reproduced.csv"
TARGET = "is_climate_sensitive"
ID_COLUMN = "ID"
THRESHOLD = 0.275
C1_WEIGHT = 0.30


def make_features(frame):
    data = frame.copy()

    if "deathdate" in data.columns:
        date = pd.to_datetime(data["deathdate"], errors="coerce")
        data["death_year"] = date.dt.year.astype("Int64").astype(str)
        data["death_dayofweek"] = date.dt.dayofweek
        data["death_dayofyear"] = date.dt.dayofyear
        data["death_month"] = date.dt.month
        data["death_quarter"] = date.dt.quarter
        data["death_is_weekend"] = date.dt.dayofweek.isin([5, 6]).astype(int)
        if {"max_temperature", "min_temperature"}.issubset(data.columns):
            data["temperature_range"] = data["max_temperature"] - data["min_temperature"]
        if "precipitation" in data.columns:
            data["is_rainy_day_current"] = (data["precipitation"] > 0).astype(int)
        data = data.drop(columns=["deathdate"])

    if {"avg_temperature", "max_temperature", "min_temperature"}.issubset(data.columns):
        data["temp_spread"] = data["max_temperature"] - data["min_temperature"]
        data["avg_temp_sq"] = data["avg_temperature"] ** 2

    if "age" in data.columns:
        data["age_sq"] = data["age"] ** 2

    if {"latitude", "longitude"}.issubset(data.columns):
        data["lat_lon_interaction"] = data["latitude"] * data["longitude"]

    if {"age", "avg_temperature"}.issubset(data.columns):
        data["age_temp_interaction"] = data["age"] * data["avg_temperature"]

    if {"rain_sum_30d", "tavg_30d"}.issubset(data.columns):
        data["rain_temp_ratio_30d"] = data["rain_sum_30d"] / (data["tavg_30d"] + 1)

    if {"rain_sum_30d", "precipitation"}.issubset(data.columns):
        data["rain_daily_ratio"] = data["rain_sum_30d"] / (data["precipitation"] + 1)

    if {"ndvi_30d", "elevation"}.issubset(data.columns):
        data["ndvi_elevation_interaction"] = data["ndvi_30d"] * data["elevation"]

    if {"tmax_30d", "tmin_30d"}.issubset(data.columns):
        data["temp_span_30d"] = data["tmax_30d"] - data["tmin_30d"]

    if {"age", "elevation"}.issubset(data.columns):
        data["age_elevation_interaction"] = data["age"] * data["elevation"]

    if {"avg_temperature", "elevation"}.issubset(data.columns):
        data["temp_elevation_interaction"] = data["avg_temperature"] * data["elevation"]

    if {"precipitation", "avg_temperature"}.issubset(data.columns):
        data["rain_temperature_index"] = data["precipitation"] / (data["avg_temperature"] + 1)

    for column in ["location", "zone", "gender"]:
        if column in data.columns and data[column].dtype == object:
            data[column] = data[column].astype(str).str.strip()
            data[column] = data[column].replace({"nan": "", "None": "", "NaN": ""})

    return data.drop(columns=[TARGET, ID_COLUMN], errors="ignore")


def make_model(c_value, numeric_columns, categorical_columns):
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric_columns),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical_columns),
    ])
    return Pipeline([
        ("features", preprocessor),
        ("classifier", LogisticRegression(
            C=c_value,
            class_weight="balanced",
            solver="liblinear",
            max_iter=8000,
            random_state=42,
        )),
    ])


def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    if CLIMATE_PATH.exists():
        climate = pd.read_csv(CLIMATE_PATH)
        climate = climate.drop(columns=["deathdate"], errors="ignore")
        climate = climate.drop_duplicates(subset=[ID_COLUMN], keep="last")
        train = train.merge(climate, on=ID_COLUMN, how="left")
        test = test.merge(climate, on=ID_COLUMN, how="left")

    target = train[TARGET].astype(int)
    features = make_features(train)
    test_features = make_features(test)

    features = features.drop(columns=["location"], errors="ignore")
    test_features = test_features.drop(columns=["location"], errors="ignore")

    for column in list(features.columns):
        if features[column].isna().all():
            features = features.drop(columns=[column])
            test_features = test_features.drop(columns=[column], errors="ignore")

    categorical_columns = features.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_columns = features.select_dtypes(exclude=["object", "category"]).columns.tolist()
    for column in categorical_columns:
        features[column] = features[column].fillna("__missing__").astype(str)
        test_features[column] = test_features[column].fillna("__missing__").astype(str)

    model_c2 = make_model(2.0, numeric_columns, categorical_columns)
    model_c1 = make_model(1.0, numeric_columns, categorical_columns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_c2.fit(features, target)
        model_c1.fit(features, target)

    probability_c2 = model_c2.predict_proba(test_features)[:, 1]
    probability_c1 = model_c1.predict_proba(test_features)[:, 1]
    probabilities = (1 - C1_WEIGHT) * probability_c2 + C1_WEIGHT * probability_c1

    submission = pd.DataFrame({
        ID_COLUMN: test[ID_COLUMN],
        "TargetF1": (probabilities >= THRESHOLD).astype(int),
        "TargetRAUC": probabilities,
    })
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {OUTPUT_PATH.name} with {len(submission)} rows")
    print(f"Blend: 70% C=2.0 + 30% C=1.0; TargetF1 threshold: {THRESHOLD}")
    print(submission.head().to_string(index=False))


if __name__ == "__main__":
    main()
