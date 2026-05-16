"""
train_model.py — src/models/train_model.py
Fixed to match actual params.yaml structure from Personne 1.
"""

import os
import yaml
import joblib
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    mean_squared_error,
    r2_score,
)
from xgboost import XGBClassifier, XGBRegressor
import optuna
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.models.signature import infer_signature

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Load params ────────────────────────────────────────────────────
with open("params.yaml", "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)

# ── Fixed: match actual params.yaml keys ──────────────────────────
SEED      = params["data"]["random_state"]           # was params["project"]["random_seed"]
MIN_ACC   = params["model"]["min_accuracy"]          # was params["classification"]["min_accuracy"]
CLASSES   = ["Plastique", "Verre", "Papier", "Métal"]  # hardcoded — not in params.yaml
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Feature columns from params.yaml ──────────────────────────────
FEATURE_COLS = params["columns"]["numeric_features"]  # all features for classification
TARGET       = params["columns"]["target"]            # Categorie
PRICE_COL    = params["columns"]["price"]             # Prix_Revente
TEXT_COL     = params["columns"]["text"]              # Rapport_Collecte


# ══════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════

def load_data():
    """Load preprocessed splits using paths from params.yaml."""
    train = pd.read_csv(params["data"]["train_path"])
    val   = pd.read_csv(params["data"]["val_path"])
    test  = pd.read_csv(params["data"]["test_path"])
    logger.info(f"Data loaded — train:{train.shape} val:{val.shape} test:{test.shape}")
    return train, val, test


def get_classification_splits(train, val, test):
    """
    X = numeric_features from params (includes Source OHE)
    y = Categorie
    Drop rows with missing target (514 unlabelled rows).
    """
    train = train.dropna(subset=[TARGET])
    val   = val.dropna(subset=[TARGET])
    test  = test.dropna(subset=[TARGET])

    # Use only columns that exist in the data
    feat = [c for c in FEATURE_COLS if c in train.columns
            and c not in [TARGET, TEXT_COL, PRICE_COL]]

    X_train = train[feat]
    X_val   = val[feat]
    X_test  = test[feat]
    y_train = train[TARGET]
    y_val   = val[TARGET]
    y_test  = test[TARGET]

    logger.info(f"Classification features: {feat}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_regression_splits(train, val, test):
    """
    X = numeric_features excluding Prix_Revente
    y = Prix_Revente (already standardized, mean≈0)
    """
    train = train.dropna(subset=[PRICE_COL])
    val   = val.dropna(subset=[PRICE_COL])
    test  = test.dropna(subset=[PRICE_COL])

    feat = [c for c in FEATURE_COLS if c in train.columns
            and c not in [TARGET, TEXT_COL, PRICE_COL]]

    X_train = train[feat]
    X_val   = val[feat]
    X_test  = test[feat]
    y_train = train[PRICE_COL]
    y_val   = val[PRICE_COL]
    y_test  = test[PRICE_COL]

    return X_train, X_val, X_test, y_train, y_val, y_test


# ══════════════════════════════════════════════════════════════════
# 2. CLASSIFICATION — 4 BASELINE MODELS
# ══════════════════════════════════════════════════════════════════

def run_classification_baselines(X_train, X_val, X_test,
                                  y_train, y_val, y_test):
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc   = le.transform(y_val)
    y_test_enc  = le.transform(y_test)

    models = {
        "RandomForest":       RandomForestClassifier(random_state=SEED),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=SEED),
        "SVM":                SVC(random_state=SEED, probability=True),
        "XGBoost":            XGBClassifier(
                                  use_label_encoder=False,
                                  eval_metric="mlogloss",
                                  random_state=SEED,
                              ),
    }

    results = {}
    for name, model in models.items():
        logger.info(f"Training {name}...")
        if name == "XGBoost":
            model.fit(X_train, y_train_enc)
            preds = model.predict(X_val)
            acc   = accuracy_score(y_val_enc, preds)
            f1    = f1_score(y_val_enc, preds, average="weighted")
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            acc   = accuracy_score(y_val, preds)
            f1    = f1_score(y_val, preds, average="weighted")

        results[name] = {"model": model, "accuracy": acc, "f1": f1}
        logger.info(f"  {name}: Accuracy={acc:.4f} F1={f1:.4f}")

    return results, le


# ══════════════════════════════════════════════════════════════════
# 3. OPTUNA TUNING
# ══════════════════════════════════════════════════════════════════

def tune_random_forest(X_train, X_val, y_train, y_val, n_trials=50):

    def objective(trial):
        clf = RandomForestClassifier(
            n_estimators      = trial.suggest_int("n_estimators", 50, 300),
            max_depth         = trial.suggest_int("max_depth", 3, 20),
            min_samples_split = trial.suggest_int("min_samples_split", 2, 10),
            min_samples_leaf  = trial.suggest_int("min_samples_leaf", 1, 5),
            max_features      = trial.suggest_categorical("max_features",
                                    ["sqrt", "log2"]),
            random_state=SEED,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        return accuracy_score(y_val, clf.predict(X_val))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    logger.info(f"Best RF params  : {study.best_params}")
    logger.info(f"Best RF accuracy: {study.best_value:.4f}")

    best_rf = RandomForestClassifier(
        **study.best_params, random_state=SEED, n_jobs=-1
    )
    best_rf.fit(X_train, y_train)
    return best_rf, study.best_params, study.best_value


# ══════════════════════════════════════════════════════════════════
# 4. STACKING
# ══════════════════════════════════════════════════════════════════

def build_stacking_classifier(le, X_train, y_train, X_test, y_test_enc):
    y_train_enc = le.transform(y_train)

    stack = StackingClassifier(
        estimators=[
            ("rf",  RandomForestClassifier(n_estimators=100, random_state=SEED)),
            ("xgb", XGBClassifier(use_label_encoder=False,
                                   eval_metric="mlogloss", random_state=SEED)),
        ],
        final_estimator=LogisticRegression(),
        cv=3,
        n_jobs=-1,
    )
    stack.fit(X_train, y_train_enc)
    acc = accuracy_score(y_test_enc, stack.predict(X_test))
    logger.info(f"Stacking accuracy (test): {acc:.4f}")
    return stack, acc


# ══════════════════════════════════════════════════════════════════
# 5. REGRESSION
# ══════════════════════════════════════════════════════════════════

def run_regression_baselines(X_train, X_val, X_test,
                              y_train, y_val, y_test):
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest":     RandomForestRegressor(random_state=SEED),
        "XGBoost":          XGBRegressor(random_state=SEED),
    }

    results = {}
    for name, model in models.items():
        logger.info(f"Training regressor {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        rmse  = np.sqrt(mean_squared_error(y_val, preds))
        r2    = r2_score(y_val, preds)
        results[name] = {"model": model, "rmse": rmse, "r2": r2}
        logger.info(f"  {name}: RMSE={rmse:.4f} R²={r2:.4f}")

    best_name = min(results, key=lambda k: results[k]["rmse"])
    logger.info(f"Best regressor: {best_name}")
    return results, best_name


# ══════════════════════════════════════════════════════════════════
# 🟢 6. MLFLOW TRACKING
# ══════════════════════════════════════════════════════════════════

def setup_mlflow():
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    logger.info(f"MLflow URI        : {params['mlflow']['tracking_uri']}")
    logger.info(f"MLflow experiment : {params['mlflow']['experiment_name']}")


def log_experiment_1_baselines(results_clf):
    with mlflow.start_run(run_name="EXP1_baseline_classifiers"):
        mlflow.set_tag("experiment_type", "classification_baseline")
        mlflow.set_tag("classes", str(CLASSES))
        mlflow.log_param("random_seed", SEED)
        mlflow.log_param("features", str(FEATURE_COLS))

        for name, res in results_clf.items():
            mlflow.log_metric(f"{name}_accuracy", res["accuracy"])
            mlflow.log_metric(f"{name}_f1",       res["f1"])

        best_name = max(results_clf, key=lambda k: results_clf[k]["accuracy"])
        mlflow.log_metric("best_baseline_accuracy", results_clf[best_name]["accuracy"])
        mlflow.log_param("best_baseline_model", best_name)
        logger.info(f"[MLflow] EXP1 logged — best: {best_name}")


def log_experiment_2_rf_optuna(best_rf, best_params, best_val_acc,
                                X_test, y_test):
    test_acc = accuracy_score(y_test, best_rf.predict(X_test))
    test_f1  = f1_score(y_test, best_rf.predict(X_test), average="weighted")

    with mlflow.start_run(run_name="EXP2_rf_optuna_tuned"):
        mlflow.set_tag("experiment_type", "classification_tuning")
        mlflow.set_tag("tuning_method", "optuna")
        mlflow.log_params(best_params)
        mlflow.log_param("random_seed", SEED)
        mlflow.log_metric("val_accuracy",  best_val_acc)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_f1",       test_f1)
        mlflow.sklearn.log_model(best_rf, "random_forest_tuned")
        logger.info(f"[MLflow] EXP2 logged — test_acc={test_acc:.4f}")
    return test_acc


def log_experiment_3_stacking(stack, le, X_test, y_test):
    y_test_enc = le.transform(y_test)
    preds      = stack.predict(X_test)
    test_acc   = accuracy_score(y_test_enc, preds)
    test_f1    = f1_score(y_test_enc, preds, average="weighted")

    with mlflow.start_run(run_name="EXP3_stacking_classifier"):
        mlflow.set_tag("experiment_type", "classification_stacking")
        mlflow.set_tag("base_models", "RandomForest + XGBoost")
        mlflow.set_tag("meta_model", "LogisticRegression")
        mlflow.log_param("cv_folds", 3)
        mlflow.log_param("random_seed", SEED)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_f1",       test_f1)
        mlflow.sklearn.log_model(stack, "stacking_classifier")
        logger.info(f"[MLflow] EXP3 logged — test_acc={test_acc:.4f}")
    return test_acc


def log_experiment_4_regression(results_reg, best_reg_name, X_test, y_test):
    best_reg   = results_reg[best_reg_name]["model"]
    test_preds = best_reg.predict(X_test)
    test_rmse  = np.sqrt(mean_squared_error(y_test, test_preds))
    test_r2    = r2_score(y_test, test_preds)

    with mlflow.start_run(run_name="EXP4_regression_comparison"):
        mlflow.set_tag("experiment_type", "regression")
        mlflow.set_tag("target", "Prix_Revente_standardized")
        mlflow.log_param("best_regressor", best_reg_name)
        mlflow.log_param("random_seed", SEED)

        for name, res in results_reg.items():
            mlflow.log_metric(f"{name}_val_rmse", res["rmse"])
            mlflow.log_metric(f"{name}_val_r2",   res["r2"])

        mlflow.log_metric("best_test_rmse", test_rmse)
        mlflow.log_metric("best_test_r2",   test_r2)
        mlflow.sklearn.log_model(best_reg, "best_regressor")
        logger.info(f"[MLflow] EXP4 logged — {best_reg_name} RMSE={test_rmse:.4f}")
    return best_reg


def log_experiment_5_final(best_rf, best_reg,
                            X_train, X_val, X_test,
                            y_train, y_val, y_test_clf, y_test_reg):
    clf_acc  = accuracy_score(y_test_clf, best_rf.predict(X_test))
    clf_f1   = f1_score(y_test_clf, best_rf.predict(X_test), average="weighted")
    reg_preds = best_reg.predict(X_test)
    reg_rmse  = np.sqrt(mean_squared_error(y_test_reg, reg_preds))
    reg_r2    = r2_score(y_test_reg, reg_preds)

    with mlflow.start_run(run_name="EXP5_final_production_models"):
        mlflow.set_tag("experiment_type", "production_candidate")
        mlflow.set_tag("status", "champion")

        mlflow.log_param("classifier",  "RandomForest_Optuna")
        mlflow.log_param("regressor",   best_reg.__class__.__name__)
        mlflow.log_param("random_seed", SEED)
        mlflow.log_param("train_size",  len(X_train))
        mlflow.log_param("min_accuracy_threshold", MIN_ACC)

        mlflow.log_metric("clf_test_accuracy", clf_acc)
        mlflow.log_metric("clf_test_f1",       clf_f1)
        mlflow.log_metric("reg_test_rmse",     reg_rmse)
        mlflow.log_metric("reg_test_r2",       reg_r2)
        mlflow.log_metric("accuracy_threshold_ok", int(clf_acc >= MIN_ACC))

        # ── Register classifier ────────────────────────────────────
        sig_clf = infer_signature(X_val, best_rf.predict(X_val))
        mlflow.sklearn.log_model(
            sk_model              = best_rf,
            artifact_path         = "classifier",
            signature             = sig_clf,
            registered_model_name = params["mlflow"]["experiment_name"],
        )

        # ── Register regressor ─────────────────────────────────────
        sig_reg = infer_signature(X_val, best_reg.predict(X_val))
        mlflow.sklearn.log_model(
            sk_model              = best_reg,
            artifact_path         = "regressor",
            signature             = sig_reg,
            registered_model_name = params["mlflow"]["experiment_name"] + "_regressor",
        )

        logger.info(f"[MLflow] EXP5 logged — clf_acc={clf_acc:.4f} reg_rmse={reg_rmse:.4f}")
        logger.info("[MLflow] Models registered ✅")

    return clf_acc, reg_rmse


# ══════════════════════════════════════════════════════════════════
# 7. SAVE MODELS
# ══════════════════════════════════════════════════════════════════

def save_models(classifier, regressor):
    clf_path = params["model"]["classifier_path"]
    reg_path = params["model"]["regressor_path"]

    Path(clf_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, clf_path)
    joblib.dump(regressor,  reg_path)

    logger.info(f"✅ Classifier → {clf_path}")
    logger.info(f"✅ Regressor  → {reg_path}")


# ══════════════════════════════════════════════════════════════════
# 8. MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 60)
    logger.info("ECO-SMART CLASSIFIER — Training Pipeline")
    logger.info("=" * 60)

    train, val, test = load_data()

    # Classification splits
    X_tr_c, X_v_c, X_te_c, y_tr_c, y_v_c, y_te_c = \
        get_classification_splits(train, val, test)

    # Regression splits
    X_tr_r, X_v_r, X_te_r, y_tr_r, y_v_r, y_te_r = \
        get_regression_splits(train, val, test)

    setup_mlflow()

    # EXP1 — baselines
    logger.info("\n[STEP 1] Baseline classifiers...")
    results_clf, le = run_classification_baselines(
        X_tr_c, X_v_c, X_te_c, y_tr_c, y_v_c, y_te_c
    )
    log_experiment_1_baselines(results_clf)

    # EXP2 — Optuna RF
    logger.info("\n[STEP 2] Optuna tuning...")
    best_rf, best_params, best_val_acc = tune_random_forest(
        X_tr_c, X_v_c, y_tr_c, y_v_c, n_trials=20  # 20 for speed, change to 50
    )
    log_experiment_2_rf_optuna(best_rf, best_params, best_val_acc, X_te_c, y_te_c)

    # EXP3 — Stacking
    logger.info("\n[STEP 3] Stacking...")
    y_te_c_enc = le.transform(y_te_c)
    stack, _ = build_stacking_classifier(le, X_tr_c, y_tr_c, X_te_c, y_te_c_enc)
    log_experiment_3_stacking(stack, le, X_te_c, y_te_c)

    # EXP4 — Regression
    logger.info("\n[STEP 4] Regression...")
    results_reg, best_reg_name = run_regression_baselines(
        X_tr_r, X_v_r, X_te_r, y_tr_r, y_v_r, y_te_r
    )
    best_reg = log_experiment_4_regression(
        results_reg, best_reg_name, X_te_r, y_te_r
    )

    # EXP5 — Final + Registry
    logger.info("\n[STEP 5] Final models + MLflow Registry...")
    clf_acc, reg_rmse = log_experiment_5_final(
        best_rf, best_reg,
        X_tr_c, X_v_c, X_te_c,
        y_tr_c, y_v_c, y_te_c,
        y_te_r
    )

    save_models(best_rf, best_reg)

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"  Classifier accuracy : {clf_acc:.4f}")
    logger.info(f"  Regressor RMSE      : {reg_rmse:.4f}")
    logger.info(f"  MLflow experiments  : 5 ✅")
    logger.info(f"  Models saved        : {params['model']['classifier_path']} ✅")
    logger.info("=" * 60)

    assert clf_acc >= MIN_ACC, \
        f"❌ Accuracy {clf_acc:.4f} below threshold {MIN_ACC}"
    logger.info(f"✅ Accuracy threshold ≥ {MIN_ACC} satisfied")


if __name__ == "__main__":
    main()