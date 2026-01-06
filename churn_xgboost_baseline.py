from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file: {path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Baseline churn model (XGBoost) on user_monthly with time-based split.")
    p.add_argument("--data-dir", type=str, required=True, help="Directory containing users.* and user_monthly.*")
    p.add_argument("--format", type=str, default="parquet", choices=["csv", "parquet"])
    p.add_argument("--split", type=str, default="user", choices=["user", "time"], help="How to split train/test.")
    p.add_argument("--test-size", type=float, default=0.2, help="For split=user only.")
    p.add_argument("--test-months", type=int, default=3, help="How many last months to use as test split.")
    p.add_argument("--seed", type=int, default=7)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = Path(args.data_dir).expanduser().resolve()
    fmt = args.format

    users_path = data_dir / f"users.{fmt}"
    monthly_path = data_dir / f"user_monthly.{fmt}"
    if not users_path.exists():
        raise FileNotFoundError(users_path)
    if not monthly_path.exists():
        raise FileNotFoundError(monthly_path)

    users = _read_table(users_path)
    m = _read_table(monthly_path)

    m["month"] = pd.to_datetime(m["month"]).dt.to_period("M").dt.to_timestamp()

    # Join static features (improves baseline & matches real-life feature sets)
    keep_users_cols = [
        "user_id",
        "country",
        "company_size",
        "industry",
        "acquisition_channel",
        "billing_period",
        "seats_purchased",
        "discount_pct",
        "payment_method",
    ]
    keep_users_cols = [c for c in keep_users_cols if c in users.columns]
    if keep_users_cols:
        m = m.merge(users[keep_users_cols], on="user_id", how="left")

    # Feature engineering (minimal, but realistic): lags + rolling stats
    m = m.sort_values(["user_id", "month"])
    for col in ["sessions", "feature_usage_score", "support_tickets", "payment_failures", "mrr"]:
        if col not in m.columns:
            continue
        g = m.groupby("user_id", observed=True)[col]
        m[f"{col}_lag1"] = g.shift(1)
        m[f"{col}_lag2"] = g.shift(2)
        m[f"{col}_delta1"] = m[col] - m[f"{col}_lag1"]
        m[f"{col}_roll3_mean"] = g.rolling(3).mean().reset_index(level=0, drop=True)

    # Label (no leakage)
    y = m["churned_next_month"].astype(int).to_numpy()

    # Features: remove leakage + identifiers
    drop_cols = {"churned", "churned_next_month", "user_id"}
    X = m.drop(columns=[c for c in drop_cols if c in m.columns])

    if args.split == "time":
        # Time split: last N months are test
        months_sorted = np.sort(X["month"].unique())
        if len(months_sorted) < args.test_months + 3:
            raise ValueError("Not enough months for a time-based split; generate a longer dataset.")
        cutoff = months_sorted[-args.test_months]
        is_test = X["month"] >= cutoff
    else:
        # User split: test users are held-out entirely (prevents leakage across months)
        rng = np.random.default_rng(args.seed)
        uids = m["user_id"].unique()
        n_test = max(1, int(round(len(uids) * float(args.test_size))))
        test_uids = set(rng.choice(uids, size=n_test, replace=False).tolist())
        is_test = m["user_id"].isin(test_uids).to_numpy()

    # Prep: drop month from features (avoid model keying on absolute time)
    X = X.drop(columns=["month"])

    # Basic preprocessing: one-hot for categoricals
    cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype) == "category"]
    X_enc = pd.get_dummies(X, columns=cat_cols, dummy_na=True)
    X_enc = X_enc.replace([np.inf, -np.inf], np.nan)
    X_enc = X_enc.fillna(0.0).astype(np.float32)

    X_train = X_enc.loc[~is_test].to_numpy()
    y_train = y[~is_test]
    X_test = X_enc.loc[is_test].to_numpy()
    y_test = y[is_test]

    # Model
    try:
        from xgboost import XGBClassifier
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            "xgboost is not installed. Install with:\n"
            "  python3 -m pip install -r requirements-ml.txt\n\n"
            f"Original import error: {e}"
        )

    pos = float(y_train.sum())
    neg = float(len(y_train) - y_train.sum())
    scale_pos_weight = float(neg / max(1.0, pos))

    model = XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        min_child_weight=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=args.seed,
        n_jobs=0,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X_train, y_train)

    p = model.predict_proba(X_test)[:, 1]

    from sklearn.metrics import roc_auc_score

    auc = float(roc_auc_score(y_test, p))
    base_rate = float(y_test.mean())
    print("BASELINE_OK")
    print(f"rows_train: {len(y_train):,} | rows_test: {len(y_test):,}")
    print(f"test_base_rate: {base_rate:.4f}")
    print(f"test_auc: {auc:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


