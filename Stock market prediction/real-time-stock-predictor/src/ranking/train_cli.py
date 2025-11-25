import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from src.ranking.model import MODELS_DIR, _build_training_data


# Default set of liquid NSE tickers for demo training
DEFAULT_TICKERS = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "KOTAKBANK.NS",
    "SBIN.NS",
    "ITC.NS",
    "AXISBANK.NS",
    "LT.NS",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the ranking model on NSE tickers",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=",".join(DEFAULT_TICKERS),
        help="Comma-separated list of NSE tickers (Yahoo symbols, e.g. RELIANCE.NS)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="^NSEI",
        help="Benchmark Yahoo symbol (default: ^NSEI for NIFTY 50)",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=5,
        help="Number of years of history to fetch (default: 5)",
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=21,
        help="Prediction horizon in trading days for outperformance label (default: 21)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of samples used for test split (default: 0.2)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="If set, run RandomizedSearchCV to tune RandomForest hyperparameters",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=20,
        help="Number of parameter settings sampled during tuning (default: 20)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="ranking_training_log.csv",
        help="CSV file name (under models/) to append training metrics",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="If set, do not save the trained model to disk",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        raise SystemExit("No tickers provided for training")

    print(f"Building training data for {len(tickers)} tickers vs benchmark {args.benchmark}...")
    X, y = _build_training_data(
        tickers=tickers,
        benchmark=args.benchmark,
        years=args.years,
        horizon_days=args.horizon_days,
    )
    print(f"Total samples: {X.shape[0]}, features: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=42,
        stratify=y,
    )

    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    base_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
    )

    if args.tune:
        param_dist = {
            "n_estimators": [100, 200, 300, 400],
            "max_depth": [3, 5, 7, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
        }
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=args.n_iter,
            scoring="roc_auc",
            n_jobs=-1,
            cv=3,
            random_state=42,
            verbose=1,
        )
        print("Running hyperparameter search (RandomizedSearchCV)...")
        search.fit(X_train, y_train)
        model = search.best_estimator_
        print("Best params:", search.best_params_)
        print(f"Best CV ROC AUC: {search.best_score_:.4f}")
    else:
        model = base_model
        print("Training ranking model...")
        model.fit(X_train, y_train)

    print("Evaluating on test set...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")

    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        print(f"Test ROC AUC: {auc:.4f}")
    except Exception:
        auc = None

    print("Classification report (test set):")
    print(classification_report(y_test, y_pred))

    models_dir: Path = MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    log_path = models_dir / args.log_file
    header = [
        "timestamp",
        "tickers",
        "benchmark",
        "years",
        "horizon_days",
        "test_size",
        "tuned",
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "test_accuracy",
        "test_roc_auc",
    ]

    params = model.get_params()
    row = [
        datetime.utcnow().isoformat(),
        ",".join(tickers),
        args.benchmark,
        args.years,
        args.horizon_days,
        args.test_size,
        bool(args.tune),
        params.get("n_estimators"),
        params.get("max_depth"),
        params.get("min_samples_split"),
        params.get("min_samples_leaf"),
        params.get("max_features"),
        acc,
        auc,
    ]

    write_header = not log_path.exists()
    with log_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)
    print(f"Appended metrics to log: {log_path}")

    if not args.no_save:
        from joblib import dump

        path = models_dir / "ranking_model.joblib"
        dump(model, path)
        print(f"Saved trained ranking model to: {path}")
        print("FastAPI /top_picks and the Streamlit UI will use this model.")
    else:
        print("Skipping model save (--no-save set)")


if __name__ == "__main__":
    main()
