"""
config.py - ver8 競馬予測システム 共通設定
=========================================
定数・ブラックリスト・特徴量リストの一元管理
"""

from pathlib import Path

# ──────────────────────────────────────────────
# パス設定
# ──────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DB_PATH    = BASE_DIR / "jra_data.db"
DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

for _d in [DATA_DIR, MODEL_DIR, OUTPUT_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# データ期間
# ──────────────────────────────────────────────
TRAIN_START_YEAR = 2016
TRAIN_END_YEAR   = 2024
TEST_START_YEAR  = 2025
TEST_END_YEAR    = 2026

# ──────────────────────────────────────────────
# 時間減衰設定
# ──────────────────────────────────────────────
DECAY_RATE = 0.85          # 1年ごとに 0.85 倍（2016年は約 0.85^8 ≒ 0.27）
DECAY_BASE_YEAR = 2024     # 重み 1.0 を与える基準年

# ──────────────────────────────────────────────
# ★ ブラックリスト（絶対に特徴量に使わない列）
#    直前確定情報 / レース結果 / 払戻情報
# ──────────────────────────────────────────────
FORBIDDEN_COLS = [
    # --- 当日直前情報（前日には確定しない）---
    "odds_tansho",      "tansho_odds",
    "tansho_ninki",     "tansho_ninkijun",  "ninki",
    "bataiju",          "zogen_sa",         "zogen_cd",
    # --- レース結果（ターゲット作成専用）---
    "kakutei_jyuni",    "kakutei_juni",     "jyuni",
    "time",             "tymei",            "record_time",
    "margin",           "chakusa",
    "corner1_jyuni",    "corner2_jyuni",    "corner3_jyuni",    "corner4_jyuni",
    "last3f",           "agari3f",
    # --- 払戻情報（EV計算専用）---
    "payout_tansho",    "tansho_payout",
    "payout_fukusho",   "fukusho_payout",
    "payout_umaren",    "umaren_payout",
    "payout_wide",      "wide_payout",
    "payout_umatan",    "umatan_payout",
    "payout_sanrenpuku","sanrenpuku_payout",
    "payout_sanrentan", "sanrentan_payout",
]

# ──────────────────────────────────────────────
# カテゴリ変数（LightGBM/CatBoost のネイティブ処理に渡す）
# ──────────────────────────────────────────────
CAT_COLS = [
    "jyo_cd",          # 競馬場コード
    "track_cd",        # コース種別（芝/ダ/障）
    "course_cd",       # 右・左・直線・内
    "baba_cd",         # 馬場状態（良・稍重・重・不良）
    "class_cd",        # レースクラス
    "sex",             # 性別
    "jockey_id",       # 騎手ID
    "trainer_id",      # 調教師ID
    "horse_id",        # 馬ID（血統登録番号）
    "sire_id",         # 父馬ID
    "dam_sire_id",     # 母父馬ID
    "waku_no",         # 枠番
]

# ──────────────────────────────────────────────
# ターゲット変数名
# ──────────────────────────────────────────────
TARGET_WIN          = "win"            # 1着フラグ (0/1)
TARGET_PLACE        = "place"          # 3着以内フラグ (0/1)
TARGET_TANSHO_RET   = "tansho_return"  # 単勝払戻（円/100円ベット）
TARGET_FUKUSHO_RET  = "fukusho_return" # 複勝払戻（円/100円ベット）

# 学習に使うターゲット
PRIMARY_TARGET   = TARGET_WIN
SECONDARY_TARGET = TARGET_PLACE

# ──────────────────────────────────────────────
# モデル・学習設定
# ──────────────────────────────────────────────
N_SPLITS     = 5      # TimeSeriesSplit の分割数
RANDOM_SEED  = 42
OPTUNA_TRIALS = 50

# LightGBM デフォルトパラメータ（Optunaでチューニング前の初期値）
LGBM_DEFAULT_PARAMS = {
    "objective":     "binary",
    "metric":        "auc",
    "boosting_type": "gbdt",
    "n_estimators":  1000,
    "learning_rate": 0.05,
    "num_leaves":    63,
    "max_depth":     -1,
    "min_child_samples": 50,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":  5,
    "reg_alpha":     0.1,
    "reg_lambda":    0.1,
    "verbose":       -1,
    "random_state":  RANDOM_SEED,
}

# ──────────────────────────────────────────────
# バックテスト設定
# ──────────────────────────────────────────────
BET_UNIT          = 100          # 1点あたり賭け金（円）
MIN_WIN_PROB      = 0.12         # 単勝買い最低予測勝率
MIN_EV_TANSHO     = 0.05         # 単勝 最低期待値（EV > 0.05 → 購入）
MIN_EV_PLACE      = 0.03         # 複勝 最低期待値
BOX_TOP_N         = 4            # ボックス買いに使う上位頭数（馬連/三連複）
NAGASHI_PIVOT_N   = 1            # 軸流しの軸頭数
UPSET_THRESHOLD   = 5            # 人気薄フィルター（何番人気以下を"荒れ"とみなすか）
ARERI_PAYOUT_LINE = 500          # 荒れフィルター: 過去払戻がこの値超なら荒れレース扱い
