import argparse
import pickle
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Extract title from Name column
    if "Name" in df.columns:
        df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
        rare_titles = ["Lady", "Countess", "Capt", "Col", "Don", "Dr",
                       "Major", "Rev", "Sir", "Jonkheer", "Dona"]
        df["Title"] = df["Title"].replace(rare_titles, "Rare")
        df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    else:
        df["Title"] = "Unknown"

    # Family size + alone flag
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"]    = (df["FamilySize"] == 1).astype(int)

    # Fare per person
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"].replace(0, 1)

    # Age group buckets
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["AgeGroup"] = pd.cut(df["Age"],
                            bins=[0, 12, 18, 35, 60, 100],
                            labels=["Child", "Teen", "Adult", "MiddleAge", "Senior"])
    return df


def preprocess(df: pd.DataFrame):
    df = engineer_features(df)

    # Sex → binary
    df["Sex_enc"] = (df["Sex"].str.lower() == "female").astype(int)

    # Embarked → ordinal
    df["Embarked"] = df["Embarked"].fillna("S")
    df["Embarked_enc"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2}).fillna(0).astype(int)

    # Title → label-encoded
    le = LabelEncoder()
    df["Title_enc"] = le.fit_transform(df["Title"].fillna("Unknown"))

    # AgeGroup → ordinal
    df["AgeGroup_enc"] = df["AgeGroup"].map(
        {"Child": 0, "Teen": 1, "Adult": 2, "MiddleAge": 3, "Senior": 4}
    ).fillna(2).astype(int)

    features = [
        "Pclass", "Sex_enc", "Age", "SibSp", "Parch",
        "Fare", "Embarked_enc", "FamilySize", "IsAlone",
        "FarePerPerson", "AgeGroup_enc", "Title_enc",
    ]
    for col in features:
        df[col] = df[col].fillna(df[col].median())

    return df[features], features


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline definition
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    """
    Three-step sklearn Pipeline:
      1. SimpleImputer   — fills remaining NaNs with median
      2. StandardScaler  — normalises all features to zero mean / unit variance
      3. RandomForest    — the actual classifier
    The whole Pipeline is what gets pickled, so loading it is one line.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
        )),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Train + evaluate + save
# ─────────────────────────────────────────────────────────────────────────────

def train(csv_path: str, output_pkl: str = "titanic_model.pkl") -> None:
    print(f"\n{'='*55}")
    print("  Titanic Survival Model — Training")
    print(f"{'='*55}\n")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows  |  columns: {list(df.columns)}\n")

    df = df.dropna(subset=["Survived"])
    y  = df["Survived"].astype(int)

    X, feature_names = preprocess(df)
    print(f"Features ({len(feature_names)}): {feature_names}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cv  = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")

    print(f"Accuracy      : {acc:.4f}")
    print(f"ROC-AUC       : {auc:.4f}")
    print(f"5-Fold CV AUC : {cv.mean():.4f} ± {cv.std():.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Died', 'Survived'])}")

    importances = dict(zip(feature_names,
                           pipe.named_steps["model"].feature_importances_))

    # ── Bundle + pickle ────────────────────────────────────────────────────
    #
    #  The pickle file stores a dict with 5 keys:
    #    "pipeline"      → the full fitted sklearn Pipeline  (load and call .predict_proba)
    #    "feature_names" → list of feature column names
    #    "importances"   → {feature: importance_score}
    #    "metrics"       → {accuracy, roc_auc, cv_auc}
    #    "train_size"    → int
    #    "test_size"     → int
    #
    bundle = {
        "pipeline":      pipe,
        "feature_names": feature_names,
        "importances":   importances,
        "metrics": {
            "accuracy": round(acc, 4),
            "roc_auc":  round(auc, 4),
            "cv_auc":   round(cv.mean(), 4),
        },
        "train_size": len(X_train),
        "test_size":  len(X_test),
    }

    with open(output_pkl, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\n✅  Saved  →  {output_pkl}")
    print(f"   Keys: {list(bundle.keys())}")
    print(f"\nTo reload in any Python script:")
    print(f"  import pickle")
    print(f"  bundle = pickle.load(open('{output_pkl}', 'rb'))")
    print(f"  prob   = bundle['pipeline'].predict_proba(X_new)[0][1]\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Titanic survival model")
    parser.add_argument("--csv",    default="titanic.csv",       help="Path to Titanic CSV")
    parser.add_argument("--output", default="titanic_model.pkl", help="Output pickle file")
    args = parser.parse_args()
    train(args.csv, args.output)