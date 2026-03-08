"""
step4_train_model.py - ver8
============================
TimeDecayWeight + TimeSeriesSplit + LightGBM/CatBoost + Optuna + Calibration

【設計原則】
- KFold等のランダムな分割は一切使用しない
- 必ず「過去で学習 → 未来で検証」のTimeSeriesSplitのみ使用
- 古い年のデータには指数関数的に小さい重みを付与（Concept Drift対策）
- カテゴリ変数はLightGBMのネイティブ処理に直接渡す（Target Encoding廃止）
- 確率値の校正（Calibration）を行いEV計算の精度を上げる

実行方法:
    python step4_train_model.py
"""

import sys
import warnings
import pickle
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
import lightgbm as lgb

warnings.filterwarnings("ignore")

# Optuna（インストール済みの場合のみ使用）
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠ Optuna未インストール。デフォルトパラメータで学習します。")

# CatBoost（任意）
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_DIR, MODEL_DIR,
    TRAIN_START_YEAR, TRAIN_END_YEAR,
    DECAY_RATE, DECAY_BASE_YEAR,
    CAT_COLS, FORBIDDEN_COLS,
    TARGET_WIN, TARGET_PLACE,
    N_SPLITS, RANDOM_SEED, OPTUNA_TRIALS,
    LGBM_DEFAULT_PARAMS,
)

# ═══════════════════════════════════════════════
# ユーティリティ
# ═══════════════════════════════════════════════

def _get_feature_cols(df: pd.DataFrame, target: str) -> list[str]:
    """学習に使う特徴量列を取得（ターゲット・内部列・禁止列を除外）"""
    exclude = set(FORBIDDEN_COLS) | {
        TARGET_WIN, TARGET_PLACE,
        "tansho_return", "fukusho_return",
        "_tansho_payout_raw", "_fukusho_payout_raw",
        "_odds_tansho_raw", "_ninki_raw",
        "race_id", "race_date",
        target,
    }
    return [c for c in df.columns if c not in exclude]


def compute_time_decay_weights(
    years: pd.Series,
    base_year: int = DECAY_BASE_YEAR,
    decay_rate: float = DECAY_RATE,
) -> np.ndarray:
    """
    時間減衰サンプルウェイトを計算する。
    base_year(2024) → weight=1.0
    base_year-1     → weight=0.85
    base_year-k     → weight=0.85^k
    """
    diff = base_year - years.values
    weights = np.power(decay_rate, np.clip(diff, 0, None))
    return weights.astype(np.float32)


# ═══════════════════════════════════════════════
# Optuna によるハイパーパラメータ探索
# ═══════════════════════════════════════════════

def _lgbm_objective(trial, X_train, y_train, w_train, X_val, y_val,
                    cat_features: list) -> float:
    """Optuna の目的関数（AUC最大化）"""
    params = {
        "objective":         "binary",
        "metric":            "auc",
        "boosting_type":     "gbdt",
        "n_estimators":      trial.suggest_int("n_estimators", 300, 2000),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 31, 255),
        "max_depth":         trial.suggest_int("max_depth", 4, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "feature_fraction":  trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq":      5,
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 1.0),
        "verbose":           -1,
        "random_state":      RANDOM_SEED,
    }

    # カテゴリ列インデックス
    cat_idx = [X_train.columns.get_loc(c)
               for c in cat_features if c in X_train.columns]

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(-1)],
        categorical_feature=cat_idx,
    )
    pred = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, pred)


def tune_lgbm(X_train, y_train, w_train, X_val, y_val,
              cat_features: list, n_trials: int = OPTUNA_TRIALS) -> dict:
    """Optuna でLightGBMのハイパーパラメータを最適化"""
    if not OPTUNA_AVAILABLE:
        return LGBM_DEFAULT_PARAMS.copy()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(
        lambda trial: _lgbm_objective(
            trial, X_train, y_train, w_train, X_val, y_val, cat_features
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    best = study.best_params
    best.update({
        "objective": "binary",
        "metric":    "auc",
        "verbose":   -1,
        "random_state": RANDOM_SEED,
    })
    print(f"  Best AUC: {study.best_value:.4f}")
    print(f"  Best params: {best}")
    return best


# ═══════════════════════════════════════════════
# TimeSeriesSplit による交差検証
# ═══════════════════════════════════════════════

def timeseries_cv(
    df_train: pd.DataFrame,
    target: str,
    cat_features: list,
    n_splits: int = N_SPLITS,
) -> tuple[list, list, dict]:
    """
    TimeSeriesSplit による交差検証。
    最後の fold の検証データをハイパーパラメータチューニングに使用する。

    Returns
    -------
    models  : 各 fold で学習したモデルリスト
    oof_pred: OOF 予測値
    best_params: チューニング済みパラメータ
    """
    print(f"\n[CV] TimeSeriesSplit (n_splits={n_splits})")

    # ソート確認（昇順であることが必須）
    df_train = df_train.sort_values("race_date").reset_index(drop=True)

    feature_cols = _get_feature_cols(df_train, target)
    X = df_train[feature_cols]
    y = df_train[target]
    years = df_train["year"]
    weights = compute_time_decay_weights(years)

    # カテゴリ列インデックス（LightGBMに渡す整数インデックス）
    cat_idx = [X.columns.get_loc(c) for c in cat_features if c in X.columns]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    models    = []
    oof_preds = np.zeros(len(X))
    best_params = None

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"\n  [Fold {fold+1}/{n_splits}]"
              f" Train: {len(tr_idx):,}  Val: {len(val_idx):,}"
              f" | Val期間: {df_train['race_date'].iloc[val_idx[0]].date()}"
              f" 〜 {df_train['race_date'].iloc[val_idx[-1]].date()}")

        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        w_tr        = weights[tr_idx]

        # 最終 fold のみ Optuna チューニングを実施
        if fold == n_splits - 1 and OPTUNA_AVAILABLE:
            print(f"  Optuna チューニング中 ({OPTUNA_TRIALS}試行)...")
            best_params = tune_lgbm(
                X_tr, y_tr, w_tr, X_val, y_val,
                cat_features=cat_features,
                n_trials=OPTUNA_TRIALS,
            )
        elif best_params is None:
            best_params = LGBM_DEFAULT_PARAMS.copy()

        # 学習
        model = lgb.LGBMClassifier(**best_params)
        model.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100, verbose=False),
                       lgb.log_evaluation(-1)],
            categorical_feature=cat_idx,
        )

        pred_val = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = pred_val

        auc = roc_auc_score(y_val, pred_val)
        ll  = log_loss(y_val, pred_val)
        print(f"    AUC={auc:.4f}  LogLoss={ll:.4f}")

        models.append(model)

    oof_valid = oof_preds[oof_preds > 0]
    y_valid   = y.values[oof_preds > 0]
    if len(oof_valid) > 0:
        oof_auc = roc_auc_score(y_valid, oof_valid)
        print(f"\n  OOF AUC: {oof_auc:.4f}")

    return models, oof_preds, best_params, feature_cols


# ═══════════════════════════════════════════════
# Calibration（確率校正）
# ═══════════════════════════════════════════════

def calibrate_model(model, X_cal, y_cal, cat_idx: list):
    """
    IsotonicRegression で確率を校正する。
    LightGBMはラッパーが必要なのでカスタム実装。
    """
    from sklearn.isotonic import IsotonicRegression

    raw_probs = model.predict_proba(X_cal)[:, 1]

    # Isotonic calibration
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(raw_probs, y_cal)

    return ir  # predict時は ir.predict(raw_probs) で校正後確率を取得


# ═══════════════════════════════════════════════
# 最終モデルの全期間学習
# ═══════════════════════════════════════════════

def train_final_model(
    df_train: pd.DataFrame,
    target: str,
    best_params: dict,
    cat_features: list,
    feature_cols: list,
) -> tuple:
    """
    チューニング済みパラメータで全訓練データを使って最終モデルを学習。
    Calibration も適用する（最後の20%を校正用に使う）。
    """
    print(f"\n[最終モデル学習] target={target}")

    df_train = df_train.sort_values("race_date").reset_index(drop=True)

    X = df_train[feature_cols]
    y = df_train[target]
    weights = compute_time_decay_weights(df_train["year"])

    cat_idx = [X.columns.get_loc(c) for c in cat_features if c in X.columns]

    # 校正用に後方20%を分割（時系列順）
    cal_size = int(len(X) * 0.2)
    X_train_fin, X_cal = X.iloc[:-cal_size], X.iloc[-cal_size:]
    y_train_fin, y_cal = y.iloc[:-cal_size], y.iloc[-cal_size:]
    w_train_fin        = weights[:-cal_size]

    model = lgb.LGBMClassifier(**best_params)
    model.fit(
        X_train_fin, y_train_fin,
        sample_weight=w_train_fin,
        eval_set=[(X_cal, y_cal)],
        callbacks=[lgb.early_stopping(100, verbose=False),
                   lgb.log_evaluation(-1)],
        categorical_feature=cat_idx,
    )

    # Calibration
    calibrator = calibrate_model(model, X_cal, y_cal, cat_idx)
    cal_probs  = calibrator.predict(model.predict_proba(X_cal)[:, 1])
    brier = brier_score_loss(y_cal, cal_probs)
    print(f"  Calibration Brier Score: {brier:.4f}")

    return model, calibrator


# ═══════════════════════════════════════════════
# 特徴量重要度の保存
# ═══════════════════════════════════════════════

def save_feature_importance(model, feature_cols: list, tag: str = "win"):
    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain":  model.booster_.feature_importance(importance_type="gain"),
        "importance_split": model.booster_.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)

    out_path = MODEL_DIR / f"feature_importance_{tag}.csv"
    imp.to_csv(out_path, index=False)
    print(f"\n  特徴量重要度保存: {out_path}")
    print(f"  Top-20:\n{imp.head(20).to_string(index=False)}")
    return imp


# ═══════════════════════════════════════════════
# メイン実行
# ═══════════════════════════════════════════════

def main():
    print("\n" + "="*60)
    print(" VER8 STEP 4: モデル学習")
    print("="*60)

    # ─── データ読み込み ───
    data_path = DATA_DIR / "training_data.parquet"
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} が見つかりません。先に step3 を実行してください。"
        )

    df = pd.read_parquet(data_path)
    print(f"データ読み込み: {df.shape}")

    # ─── 訓練 / テスト分割 ───
    df_train = df[df["year"].between(TRAIN_START_YEAR, TRAIN_END_YEAR)].copy()
    df_test  = df[df["year"] >= 2025].copy()
    print(f"  Train: {len(df_train):,} 行  ({TRAIN_START_YEAR}〜{TRAIN_END_YEAR}年)")
    print(f"  Test : {len(df_test):,} 行  (2025年〜)")

    # ─── カテゴリ列（存在するもののみ）───
    cat_features = [c for c in CAT_COLS if c in df_train.columns]

    results = {}

    for target in [TARGET_WIN, TARGET_PLACE]:
        print(f"\n{'─'*60}")
        print(f"ターゲット: {target}")
        print(f"{'─'*60}")

        # 交差検証 + チューニング
        models, oof_preds, best_params, feature_cols = timeseries_cv(
            df_train, target, cat_features
        )

        # 最終モデル学習
        final_model, calibrator = train_final_model(
            df_train, target, best_params, cat_features, feature_cols
        )

        # 特徴量重要度
        save_feature_importance(final_model, feature_cols, tag=target)

        # モデル保存
        model_path = MODEL_DIR / f"lgbm_{target}.pkl"
        calib_path = MODEL_DIR / f"calibrator_{target}.pkl"
        feat_path  = MODEL_DIR / f"feature_cols_{target}.json"

        with open(model_path, "wb") as f:
            pickle.dump(final_model, f)
        with open(calib_path, "wb") as f:
            pickle.dump(calibrator, f)
        with open(feat_path, "w", encoding="utf-8") as f:
            json.dump(feature_cols, f, ensure_ascii=False, indent=2)

        print(f"\n  モデル保存: {model_path}")
        print(f"  校正器保存: {calib_path}")

        # テストデータへの予測（参考）
        X_test = df_test[feature_cols]
        raw_pred_test = final_model.predict_proba(X_test)[:, 1]
        cal_pred_test = calibrator.predict(raw_pred_test)
        df_test[f"pred_{target}"]     = raw_pred_test
        df_test[f"pred_{target}_cal"] = cal_pred_test

        results[target] = {
            "best_params":   best_params,
            "feature_cols":  feature_cols,
            "model_path":    str(model_path),
        }

        # OOF 保存
        oof_path = MODEL_DIR / f"oof_preds_{target}.npy"
        np.save(oof_path, oof_preds)

    # テスト予測保存（step5 で使用）
    pred_cols = ["race_id", "race_date", "year", "uma_no"]
    if "horse_id" in df_test.columns:
        pred_cols.append("horse_id")
    pred_cols += [c for c in df_test.columns if c.startswith("pred_")]
    # バックテスト用の払戻・オッズも含める
    for extra in ["_tansho_payout_raw", "_fukusho_payout_raw",
                   "_odds_tansho_raw", "_ninki_raw",
                   TARGET_WIN, TARGET_PLACE]:
        if extra in df_test.columns:
            pred_cols.append(extra)

    pred_path = MODEL_DIR / "test_predictions.parquet"
    df_test[list(dict.fromkeys(pred_cols))].to_parquet(pred_path, index=False)
    print(f"\n✅ テスト予測保存: {pred_path}")

    print("\n" + "="*60)
    print("✅ STEP 4 完了")
    print("="*60)

    return results


if __name__ == "__main__":
    results = main()
