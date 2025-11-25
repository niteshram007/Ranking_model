import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier


MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def _download_price_series(symbol: str, period: str) -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval="1d", progress=False)
    if df is None or df.empty or "Close" not in df.columns:
        raise ValueError(f"No price data for {symbol}")
    df = df.dropna(subset=["Close"]).copy()
    df["ret"] = df["Close"].pct_change()
    return df


def _join_asset_benchmark(asset: pd.DataFrame, bench: pd.DataFrame) -> pd.DataFrame:
    a = asset[["Close", "ret"]].rename(columns={"Close": "Close_asset", "ret": "ret_asset"})
    b = bench[["Close", "ret"]].rename(columns={"Close": "Close_bench", "ret": "ret_bench"})
    df = a.join(b, how="inner")
    df = df.dropna().reset_index(drop=True)
    return df


def _build_feature_vector(df: pd.DataFrame, i: int) -> np.ndarray:
    if i < 63:
        raise ValueError("Not enough history for features")
    w21 = df.iloc[i - 20 : i + 1]
    w63 = df.iloc[i - 62 : i + 1]

    close_a_21 = w21["Close_asset"]
    close_a_63 = w63["Close_asset"]
    close_b_63 = w63["Close_bench"]

    ret_a_21 = float(close_a_21.iloc[-1] / close_a_21.iloc[0] - 1.0)
    ret_a_63 = float(close_a_63.iloc[-1] / close_a_63.iloc[0] - 1.0)

    vol_a_21 = float(w21["ret_asset"].std() * np.sqrt(252.0)) if not np.isnan(w21["ret_asset"].std()) else 0.0
    vol_a_63 = float(w63["ret_asset"].std() * np.sqrt(252.0)) if not np.isnan(w63["ret_asset"].std()) else 0.0

    ret_b_63 = float(close_b_63.iloc[-1] / close_b_63.iloc[0] - 1.0)
    rel_63 = ret_a_63 - ret_b_63

    return np.asarray([ret_a_21, ret_a_63, vol_a_21, vol_a_63, rel_63], dtype=np.float32)


def _build_training_data(
    tickers: Sequence[str], benchmark: str, years: int, horizon_days: int
) -> Tuple[np.ndarray, np.ndarray]:
    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    bench = _download_price_series(benchmark, f"{years}y")

    for symbol in tickers:
        try:
            asset = _download_price_series(symbol, f"{years}y")
        except Exception:
            continue

        df = _join_asset_benchmark(asset, bench)
        if len(df) < 252 + horizon_days:
            continue

        for i in range(252, len(df) - horizon_days):
            try:
                x = _build_feature_vector(df, i)
            except Exception:
                continue

            future = df.iloc[i + 1 : i + 1 + horizon_days]
            if len(future) < horizon_days:
                continue

            fa = float(future["Close_asset"].iloc[-1] / future["Close_asset"].iloc[0] - 1.0)
            fb = float(future["Close_bench"].iloc[-1] / future["Close_bench"].iloc[0] - 1.0)
            label = 1 if fa > fb else 0

            X_list.append(x)
            y_list.append(label)

    if not X_list:
        raise ValueError("No training data built for ranking model")

    X = np.vstack(X_list)
    y = np.asarray(y_list, dtype=np.int32)
    return X, y


def train_ranking_model(
    tickers: Sequence[str],
    benchmark: str = "^NSEI",
    years: int = 5,
    horizon_days: int = 21,
    models_dir: Path = MODELS_DIR,
) -> RandomForestClassifier:
    X, y = _build_training_data(tickers, benchmark, years, horizon_days)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / "ranking_model.joblib"
    joblib.dump(model, path)
    return model


def _compute_metrics_for_inference(df: pd.DataFrame) -> Dict[str, Any]:
    if len(df) < 30:
        raise ValueError("Not enough data for metrics")

    hist = df.tail(min(252, len(df))).copy()
    r = hist["ret_asset"].dropna()
    if r.empty:
        raise ValueError("No returns for metrics")

    cum = (1.0 + r).cumprod() - 1.0
    total_ret = float(cum.iloc[-1])
    mean_ret = float(r.mean()) if not np.isnan(r.mean()) else 0.0
    vol = float(r.std()) if not np.isnan(r.std()) else 0.0

    sharpe = 0.0
    if vol > 0:
        sharpe = mean_ret / vol * np.sqrt(252.0)

    window = min(63, len(hist) - 1)
    if window > 1:
        recent = hist["Close_asset"].iloc[-window:]
        recent_mom = float(recent.iloc[-1] / recent.iloc[0] - 1.0)
    else:
        recent_mom = 0.0

    vol_annual = vol * np.sqrt(252.0) if vol > 0 else 0.0

    return {
        "TotalReturn_1Y_%": round(total_ret * 100.0, 2),
        "Momentum_3M_%": round(recent_mom * 100.0, 2),
        "Volatility_1Y_%": round(vol_annual * 100.0, 2),
        "SharpeApprox": round(sharpe, 2),
    }


def _prepare_inference_frame(symbol: str, benchmark: str, period: str) -> pd.DataFrame:
    asset = _download_price_series(symbol, period)
    bench = _download_price_series(benchmark, period)
    df = _join_asset_benchmark(asset, bench)
    if len(df) < 63:
        raise ValueError("Not enough history for inference")
    return df


def compute_inference_features(
    symbol: str, benchmark: str, period: str = "2y"
) -> Tuple[np.ndarray, Dict[str, Any]]:
    df = _prepare_inference_frame(symbol, benchmark, period)
    i = len(df) - 1
    x = _build_feature_vector(df, i)
    metrics = _compute_metrics_for_inference(df)
    return x, metrics


class RankingService:
    def __init__(self, models_dir: Path = MODELS_DIR, cache_ttl: int = 900, default_benchmark: str = "^NSEI") -> None:
        self.models_dir = models_dir
        self.cache_ttl = cache_ttl
        self.default_benchmark = default_benchmark
        self.model: Any = None
        self._cache: Dict[Any, Tuple[float, List[Dict[str, Any]]]] = {}
        self._load_model()

    def _load_model(self) -> None:
        path = self.models_dir / "ranking_model.joblib"
        if path.exists():
            self.model = joblib.load(path)

    def is_ready(self) -> bool:
        return self.model is not None

    def refresh(self) -> None:
        self._load_model()

    def _cache_key(
        self,
        tickers: Sequence[str],
        categories: Sequence[str] | None,
        benchmark: str,
        top_k: int,
    ) -> Tuple[Any, ...]:
        return (tuple(tickers), tuple(categories) if categories is not None else None, benchmark, top_k)

    def top_picks(
        self,
        tickers: Sequence[str],
        categories: Sequence[str] | None = None,
        benchmark: str | None = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        if not self.is_ready():
            raise RuntimeError("Ranking model not available")

        if not tickers:
            return []

        if benchmark is None:
            benchmark = self.default_benchmark

        key = self._cache_key(tickers, categories, benchmark, top_k)
        now = time.time()
        if key in self._cache:
            ts, cached = self._cache[key]
            if now - ts < self.cache_ttl:
                return cached

        rows: List[Dict[str, Any]] = []
        for idx, symbol in enumerate(tickers):
            try:
                x, metrics = compute_inference_features(symbol, benchmark)
            except Exception:
                continue

            try:
                proba = float(self.model.predict_proba(x.reshape(1, -1))[0, 1])
            except Exception:
                proba = 0.5

            row: Dict[str, Any] = {
                "Ticker": symbol,
                "Category": categories[idx] if categories and idx < len(categories) else "Asset",
            }
            row.update(metrics)
            row["ProbOutperform"] = round(proba, 4)
            row["Score"] = round(proba, 4)
            rows.append(row)

        rows.sort(key=lambda r: r["Score"], reverse=True)
        if top_k > 0:
            rows = rows[:top_k]

        self._cache[key] = (now, rows)
        return rows
