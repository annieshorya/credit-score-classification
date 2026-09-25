"""
Credit score classification with XGBoost + SMOTE + GridSearchCV.

Predicts a customer's credit score band (Poor / Standard / Good) from monthly
financial records. Compares a plain XGBoost baseline against a tuned
SMOTE + XGBoost pipeline and writes metrics and plots to results/.

Usage (from the repo root):
    python src/train_xgboost.py              # full grid search (~20-30 min on 2 CPU cores)
    python src/train_xgboost.py --quick      # small grid, a few minutes
    python src/train_xgboost.py --split random   # row-level split, for comparison

Why the split matters: the dataset has up to 8 monthly rows per customer.
A random row-level split puts the same customer in train AND test, which
inflates accuracy. The default here holds out whole customers
(GroupShuffleSplit on Customer_ID) so the test score reflects unseen people.
Pass --split random to reproduce the (leaky) row-level split used in the
original coursework, for comparison.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "clean_train.csv"
RESULTS = ROOT / "results"
SEED = 42

CLASSES = ["Poor", "Standard", "Good"]  # encoded as 0, 1, 2
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August"]
ONE_HOT = ["Occupation", "Payment_Behaviour"]


def load_data(path: Path):
    df = pd.read_csv(path)
    y = df["Credit_Score"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    groups = df["Customer_ID"].to_numpy()

    X = df.drop(columns=["Credit_Score", "Customer_ID"]).copy()
    # ordinal / binary encodings for columns with a natural order
    X["Month"] = X["Month"].map({m: i + 1 for i, m in enumerate(MONTHS)})
    X["Credit_Mix"] = X["Credit_Mix"].map({"Bad": 0, "Standard": 1, "Good": 2})
    X["Payment_of_Min_Amount"] = (X["Payment_of_Min_Amount"] == "Yes").astype(int)
    return X, y, groups


def split(X, y, groups, how: str):
    if how == "group":
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        tr, te = next(gss.split(X, y, groups))
    else:
        idx = np.arange(len(y))
        tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
    return tr, te


def make_pipeline(use_smote: bool, **xgb_params):
    pre = ColumnTransformer(
        [("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ONE_HOT)],
        remainder="passthrough",
    )
    model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=SEED,
        **xgb_params,
    )
    steps = [("prep", pre)]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=SEED)))
    steps.append(("xgb", model))
    return Pipeline(steps)


def evaluate(name, pipe, X_te, y_te):
    proba = pipe.predict_proba(X_te)
    pred = proba.argmax(axis=1)
    recalls = recall_score(y_te, pred, average=None)
    return {
        "model": name,
        "accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "macro_f1": round(float(f1_score(y_te, pred, average="macro")), 4),
        "roc_auc_ovr_macro": round(float(roc_auc_score(y_te, proba, multi_class="ovr", average="macro")), 4),
        "recall_per_class": {c: round(float(r), 4) for c, r in zip(CLASSES, recalls)},
        "report": classification_report(y_te, pred, target_names=CLASSES, output_dict=True),
    }


def save_plots(pipe, X_te, y_te, tag):
    RESULTS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_estimator(
        pipe, X_te, y_te, display_labels=CLASSES, normalize="true", values_format=".2f", cmap="Blues", ax=ax
    )
    ax.set_title("Confusion matrix (row-normalised)")
    fig.tight_layout()
    fig.savefig(RESULTS / f"confusion_matrix_{tag}.png", dpi=120)
    plt.close(fig)

    proba = pipe.predict_proba(X_te)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for i, c in enumerate(CLASSES):
        RocCurveDisplay.from_predictions((y_te == i).astype(int), proba[:, i], name=c, ax=ax)
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_title("ROC curves (one-vs-rest)")
    fig.tight_layout()
    fig.savefig(RESULTS / f"roc_curves_{tag}.png", dpi=120)
    plt.close(fig)

    # top feature importances (gain) from the fitted booster
    names = pipe.named_steps["prep"].get_feature_names_out()
    names = [n.split("__", 1)[-1].replace("Payment_Behaviour_", "Spend: ").replace("Occupation_", "Job: ") for n in names]
    imp = pd.Series(pipe.named_steps["xgb"].feature_importances_, index=names).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    imp.plot.barh(ax=ax, color="#4C78A8")
    ax.set_title("Top 12 features by XGBoost importance")
    fig.tight_layout()
    fig.savefig(RESULTS / f"feature_importance_{tag}.png", dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["group", "random"], default="group")
    ap.add_argument("--quick", action="store_true", help="small grid for a fast run")
    ap.add_argument(
        "--best-from",
        type=Path,
        help="skip the grid search and refit the best params stored in a metrics JSON from an earlier run",
    )
    args = ap.parse_args()

    X, y, groups = load_data(DATA)
    tr, te = split(X, y, groups, args.split)
    X_tr, X_te, y_tr, y_te, g_tr = X.iloc[tr], X.iloc[te], y[tr], y[te], groups[tr]
    print(f"rows: train={len(tr):,} test={len(te):,} | customers: {len(set(groups)):,} | split={args.split}")

    results = {"split": args.split, "n_train": int(len(tr)), "n_test": int(len(te))}

    # 1) baseline: XGBoost with sensible defaults, no resampling
    t0 = time.time()
    base = make_pipeline(use_smote=False, n_estimators=300, max_depth=6, learning_rate=0.1)
    base.fit(X_tr, y_tr)
    results["baseline"] = evaluate("XGBoost (no SMOTE, default params)", base, X_te, y_te)
    print(f"baseline done in {time.time() - t0:.0f}s:", {k: v for k, v in results["baseline"].items() if k != "report"})

    # 2) SMOTE + XGBoost, tuned with cross-validated grid search (macro-F1)
    if args.quick:
        grid = {"xgb__max_depth": [6, 8], "xgb__learning_rate": [0.1], "xgb__n_estimators": [300]}
    else:
        grid = {
            "xgb__max_depth": [4, 6, 8],
            "xgb__learning_rate": [0.05, 0.1],
            "xgb__n_estimators": [200, 400],
            "xgb__subsample": [0.8, 1.0],
            "xgb__colsample_bytree": [0.8, 1.0],
        }
    if args.best_from:
        best = json.loads(Path(args.best_from).read_text())["grid_search"]["best_params"]
        grid = {f"xgb__{k}": [v] for k, v in best.items()}
    n_combos = int(np.prod([len(v) for v in grid.values()]))
    cv = (
        StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
        if args.split == "group"
        else StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    )
    search = GridSearchCV(
        make_pipeline(use_smote=True), grid, scoring="f1_macro", cv=cv, n_jobs=1, verbose=1, refit=True
    )
    t0 = time.time()
    search.fit(X_tr, y_tr, groups=g_tr if args.split == "group" else None)
    tuned = search.best_estimator_
    results["tuned"] = evaluate("SMOTE + XGBoost (GridSearchCV)", tuned, X_te, y_te)
    results["grid_search"] = {
        "parameter_combinations": n_combos,
        "cv_folds": 3,
        "total_fits": n_combos * 3,
        "best_params": {k.replace("xgb__", ""): (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()},
        "best_cv_macro_f1": round(float(search.best_score_), 4),
        "minutes": round((time.time() - t0) / 60, 1),
    }
    print("tuned:", {k: v for k, v in results["tuned"].items() if k != "report"})
    print("grid:", results["grid_search"])

    save_plots(tuned, X_te, y_te, args.split)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"metrics_{args.split}_split.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
