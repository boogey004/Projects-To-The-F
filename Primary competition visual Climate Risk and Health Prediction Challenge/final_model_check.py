import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
from sklearn.metrics import f1_score, roc_auc_score

TRAIN_PATH = "Train.csv"
TEST_PATH = "Test.csv" if os.path.exists("Test.csv") else "test.csv"
OUTPUT_PATH = "best_submission_v4.csv"
TARGET = "is_climate_sensitive"
ID_COLUMN = "ID"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

climate_candidates = [
    "climate_features.csv",
    "climate_features_rich.csv",
    "climate_features_rich_v1.csv",
]
climate_path = next((c for c in climate_candidates if os.path.exists(c)), None)
if climate_path is not None:
    climate_features = pd.read_csv(climate_path)
    if ID_COLUMN not in climate_features.columns and "ID" in climate_features.columns:
        climate_features = climate_features.rename(columns={"ID": ID_COLUMN})
    climate_features = climate_features.drop_duplicates(subset=[ID_COLUMN], keep="last")
    train = train.merge(climate_features, on=ID_COLUMN, how="left")
    test = test.merge(climate_features, on=ID_COLUMN, how="left")
    print(f"Loaded climate features from {climate_path}")
else:
    print("WARNING: no climate feature file found")


def make_features(frame):
    data = frame.copy()

    if "deathdate" in data.columns:
        date = pd.to_datetime(data["deathdate"], errors="coerce")
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

    for col in ["location", "zone", "gender"]:
        if col in data.columns and data[col].dtype == object:
            data[col] = data[col].astype(str).str.strip()
            data[col] = data[col].replace({"nan": "", "None": "", "NaN": ""})

    for col in data.select_dtypes(include=[np.number]).columns:
        if np.isclose(data[col].nunique(dropna=True), 1):
            data = data.drop(columns=[col])

    return data.drop(columns=[TARGET, ID_COLUMN], errors="ignore")


X = make_features(train)
y = train[TARGET].astype(int)
X_test = make_features(test)

for col in ["location"]:
    X = X.drop(columns=[col], errors="ignore")
    X_test = X_test.drop(columns=[col], errors="ignore")

for col in list(X.columns):
    if X[col].isna().all():
        X = X.drop(columns=[col])
        X_test = X_test.drop(columns=[col], errors="ignore")

categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ]), numeric),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), categorical),
])

models = []
for C in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]:
    models.append((f"logreg_C{C}", Pipeline([
        ("features", preprocessor),
        ("classifier", LogisticRegression(
            C=C,
            class_weight="balanced",
            solver="liblinear",
            max_iter=8000,
            random_state=42,
        ))
    ])))

models.append(("rf", Pipeline([
    ("features", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=1200,
        min_samples_leaf=1,
        max_depth=None,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
    ))
])))

models.append(("extra_trees", Pipeline([
    ("features", preprocessor),
    ("classifier", ExtraTreesClassifier(
        n_estimators=1200,
        max_features="sqrt",
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
    ))
])))

models.append(("gboost", Pipeline([
    ("features", preprocessor),
    ("classifier", GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.04,
        max_depth=3,
        subsample=0.9,
        random_state=42,
    ))
])))

models.append(("hgb", Pipeline([
    ("features", preprocessor),
    ("classifier", HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_depth=8,
        min_samples_leaf=15,
        max_leaf_nodes=31,
        random_state=42,
    ))
])))

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

thresholds = np.linspace(0.10, 0.90, 161)
best_model_name = None
best_model = None
best_threshold = 0.5
best_score = -1.0
best_valid_f1 = 0.0
best_valid_auc = 0.0

for model_name, model in models:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)

    valid_probability = model.predict_proba(X_valid)[:, 1]
    valid_auc = roc_auc_score(y_valid, valid_probability)

    def competition_score(threshold, proba=valid_probability, auc=valid_auc):
        prediction = (proba >= threshold).astype(int)
        return 0.60 * f1_score(y_valid, prediction) + 0.40 * auc

    current_threshold = max(thresholds, key=competition_score)
    current_f1 = f1_score(y_valid, (valid_probability >= current_threshold).astype(int))
    current_score = competition_score(current_threshold)
    print(f"{model_name}: F1={current_f1:.4f}, AUC={valid_auc:.4f}, Score={current_score:.4f}, Threshold={current_threshold:.3f}")

    if current_score > best_score:
        best_score = current_score
        best_model_name = model_name
        best_model = model
        best_threshold = current_threshold
        best_valid_f1 = current_f1
        best_valid_auc = valid_auc

print(f"BEST_MODEL={best_model_name}")
print(f"Valid F1: {best_valid_f1:.4f}")
print(f"Valid ROC AUC: {best_valid_auc:.4f}")
print(f"Competition score: {best_score:.4f}")
print(f"Selected threshold: {best_threshold:.3f}")

best_model.fit(X, y)
final_probs = best_model.predict_proba(X_test)[:, 1]
submission = pd.DataFrame({
    ID_COLUMN: test[ID_COLUMN],
    "TargetF1": (final_probs >= best_threshold).astype(int),
    "TargetRAUC": final_probs,
})
submission.to_csv(OUTPUT_PATH, index=False)
print(f"Saved {OUTPUT_PATH} with {len(submission)} rows")
print(submission.head().to_string(index=False))
