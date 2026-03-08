"""
step3_create_training_data.py - ver8
=====================================
DBからデータを読み込み、ターゲット変数と特徴量を作成する。

【設計原則】
- 時系列リーク完全排除：馬・騎手・調教師の過去実績は shift(1) + rolling で計算
- ブラックリスト列の強制削除（安全装置）
- カテゴリ変数は category 型に変換し LightGBM/CatBoost ネイティブ処理へ渡す
- Target Encoding は一切使用しない

実行方法:
    python step3_create_training_data.py
"""

import sqlite3
import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─── config のインポート ───────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DB_PATH, DATA_DIR, FORBIDDEN_COLS, CAT_COLS,
    TRAIN_START_YEAR, TEST_END_YEAR,
    TARGET_WIN, TARGET_PLACE, TARGET_TANSHO_RET, TARGET_FUKUSHO_RET,
)

# ═══════════════════════════════════════════════
# ステップ 1: DB読み込み
# ═══════════════════════════════════════════════

def load_from_db(db_path: Path) -> pd.DataFrame:
    """race_horse_merged テーブルを SQLite から読み込む"""
    print(f"\n{'='*60}")
    print(f"[STEP 3-1] DB読み込み: {db_path}")
    print(f"{'='*60}")

    if not db_path.exists():
        raise FileNotFoundError(
            f"DBが見つかりません: {db_path}\n"
            "jra_data.db を ver8/ フォルダに置いてください。"
        )

    con = sqlite3.connect(db_path)

    # テーブル存在確認
    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table'", con
    )["name"].tolist()
    print(f"  利用可能テーブル: {tables}")

    table = "race_horse_merged"
    if table not in tables:
        con.close()
        raise ValueError(
            f"テーブル '{table}' が存在しません。\n"
            f"存在するテーブル: {tables}"
        )

    # カラム情報表示
    col_info = pd.read_sql(f"PRAGMA table_info({table})", con)
    print(f"\n  テーブルカラム ({len(col_info)}列):")
    for _, row in col_info.iterrows():
        print(f"    [{int(row['cid']):>3}] {row['name']:<30} ({row['type']})")

    df = pd.read_sql(f"SELECT * FROM {table}", con)
    con.close()

    print(f"\n  読み込み完了: {len(df):,} 行 × {len(df.columns)} 列")
    return df


# ═══════════════════════════════════════════════
# ステップ 2: 基本前処理
# ═══════════════════════════════════════════════

def _detect_col(df: pd.DataFrame, candidates: list) -> str | None:
    """候補リストから存在する列名を返す"""
    return next((c for c in candidates if c in df.columns), None)


def basic_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """日付変換・race_id統一・年フィルタ・ソート"""
    print(f"\n[STEP 3-2] 基本前処理")

    # ─── 日付列の特定と変換 ───
    date_col = _detect_col(df, [
        "race_date", "date", "kaisai_date", "hdate", "DATE", "kai_date"
    ])
    if date_col:
        df["race_date"] = pd.to_datetime(df[date_col], errors="coerce")
    elif all(c in df.columns for c in ["year", "month", "day"]):
        df["race_date"] = pd.to_datetime(
            df["year"].astype(str) + "-" +
            df["month"].astype(str).str.zfill(2) + "-" +
            df["day"].astype(str).str.zfill(2),
            errors="coerce"
        )
    else:
        raise ValueError(
            "日付列が見つかりません。"
            "'race_date' または 'year/month/day' 列が必要です。"
        )

    # 日付派生
    df["year"]        = df["race_date"].dt.year
    df["month"]       = df["race_date"].dt.month
    df["day_of_week"] = df["race_date"].dt.dayofweek  # 0=月, 6=日

    # ─── race_id の統一 ───
    if "race_id" not in df.columns:
        id_parts = [
            _detect_col(df, ["jyo_cd", "JYO_CD"]),
            "year",
            _detect_col(df, ["kaiji", "kai"]),
            _detect_col(df, ["nichiji", "nichi"]),
            _detect_col(df, ["race_no", "r", "race_num"]),
        ]
        id_parts = [p for p in id_parts if p and p in df.columns]
        if len(id_parts) >= 3:
            df["race_id"] = df[id_parts].astype(str).apply("_".join, axis=1)
        else:
            df["race_id"] = (
                df["race_date"].dt.strftime("%Y%m%d")
                + "_" + df.get("jyo_cd", "00").astype(str)
                + "_" + df.index.astype(str)
            )

    # ─── uma_no (馬番) ───
    uma_col = _detect_col(df, ["uma_no", "UMA_NO", "horse_no", "umano"])
    if uma_col and uma_col != "uma_no":
        df["uma_no"] = df[uma_col]
    elif "uma_no" not in df.columns:
        df["uma_no"] = 0

    # ─── horse_id ───
    horse_col = _detect_col(df, [
        "horse_id", "ketto_toroku_bango", "uma_id",
        "KETTO_TOROKU_BANGO", "horse_reg_no"
    ])
    if horse_col and horse_col != "horse_id":
        df["horse_id"] = df[horse_col].astype(str)

    # ─── jockey_id ───
    jockey_col = _detect_col(df, [
        "jockey_id", "kisyu_cd", "kishi_cd", "jockeyid"
    ])
    if jockey_col and jockey_col != "jockey_id":
        df["jockey_id"] = df[jockey_col].astype(str)

    # ─── trainer_id ───
    trainer_col = _detect_col(df, [
        "trainer_id", "chokyosi_cd", "trainer_cd"
    ])
    if trainer_col and trainer_col != "trainer_id":
        df["trainer_id"] = df[trainer_col].astype(str)

    # ─── 出走頭数 ───
    df["field_size"] = df.groupby("race_id")["race_id"].transform("count")

    # ─── 年フィルタ ───
    n_before = len(df)
    df = df[df["year"].between(TRAIN_START_YEAR, TEST_END_YEAR)].reset_index(drop=True)
    print(f"  年フィルタ: {n_before:,} → {len(df):,} 行"
          f" ({df['year'].min()}〜{df['year'].max()}年)")

    # ─── ソート ───
    df = df.sort_values(["race_date", "race_id", "uma_no"]).reset_index(drop=True)
    print(f"  前処理完了: {df.shape}")
    return df


# ═══════════════════════════════════════════════
# ステップ 3: ターゲット変数の作成
# ═══════════════════════════════════════════════

def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    勝ちフラグ・複勝フラグ・払戻ベースのターゲットを作成。
    odds や payout はターゲット計算にのみ使用し、特徴量には残さない。
    """
    print(f"\n[STEP 3-3] ターゲット変数作成")

    # 着順列
    jyuni_col = _detect_col(df, [
        "kakutei_jyuni", "kakutei_juni", "jyuni",
        "finish_pos", "finish", "chaku_jyuni"
    ])
    if jyuni_col is None:
        raise ValueError(
            "着順列が見つかりません。"
            "'kakutei_jyuni' または 'jyuni' 列が必要です。"
        )
    df["_jyuni"] = pd.to_numeric(df[jyuni_col], errors="coerce")

    # 1着・3着以内フラグ
    df[TARGET_WIN]   = (df["_jyuni"] == 1).astype(np.int8)
    df[TARGET_PLACE] = (df["_jyuni"] <= 3).astype(np.int8)

    # 単勝払戻
    ptan_col = _detect_col(df, [
        "payout_tansho", "tansho_payout", "pay_tansho",
        "haraimodoshi_tansho", "tan_payout"
    ])
    if ptan_col:
        pay_tan = pd.to_numeric(df[ptan_col], errors="coerce")
        df[TARGET_TANSHO_RET]    = df[TARGET_WIN] * pay_tan
        df["_tansho_payout_raw"] = pay_tan          # バックテスト用に保存
    else:
        print("  ⚠ 単勝払戻列が見つかりません。単勝ターゲットを 0 で埋めます。")
        df[TARGET_TANSHO_RET]    = 0.0
        df["_tansho_payout_raw"] = np.nan

    # 複勝払戻
    pfuku_col = _detect_col(df, [
        "payout_fukusho", "fukusho_payout", "pay_fukusho",
        "haraimodoshi_fukusho", "fuku_payout"
    ])
    if pfuku_col:
        pay_fuku = pd.to_numeric(df[pfuku_col], errors="coerce")
        df[TARGET_FUKUSHO_RET]    = df[TARGET_PLACE] * pay_fuku
        df["_fukusho_payout_raw"] = pay_fuku
    else:
        df[TARGET_FUKUSHO_RET]    = 0.0
        df["_fukusho_payout_raw"] = np.nan

    # 単勝オッズ（EV計算用・バックテスト用に保持。特徴量ではない）
    odds_col = _detect_col(df, [
        "odds_tansho", "tansho_odds", "tan_odds"
    ])
    if odds_col:
        df["_odds_tansho_raw"] = pd.to_numeric(df[odds_col], errors="coerce")
    else:
        df["_odds_tansho_raw"] = np.nan

    # 人気順（バックテスト参照用に保持。特徴量ではない）
    ninki_col = _detect_col(df, [
        "tansho_ninki", "ninki", "tansho_ninkijun"
    ])
    if ninki_col:
        df["_ninki_raw"] = pd.to_numeric(df[ninki_col], errors="coerce")
    else:
        df["_ninki_raw"] = np.nan

    win_rate   = df[TARGET_WIN].mean()
    place_rate = df[TARGET_PLACE].mean()
    print(f"  勝率: {win_rate:.3f}  複勝率: {place_rate:.3f}")
    print(f"  単勝払戻 非null率: {df[TARGET_TANSHO_RET].notna().mean():.2%}")
    return df


# ═══════════════════════════════════════════════
# ステップ 4: 特徴量エンジニアリング（時系列リークなし）
# ═══════════════════════════════════════════════

def _rolling_shift(series: pd.Series, n: int, agg: str = "mean") -> pd.Series:
    """
    shift(1) → rolling(n) による時系列リークなし集計。
    shift(1) で当レース自身を除外してから rolling する。
    """
    shifted = series.shift(1)
    if agg == "mean":
        return shifted.rolling(n, min_periods=1).mean()
    elif agg == "max":
        return shifted.rolling(n, min_periods=1).max()
    elif agg == "min":
        return shifted.rolling(n, min_periods=1).min()
    elif agg == "std":
        return shifted.rolling(n, min_periods=1).std()
    elif agg == "sum":
        return shifted.rolling(n, min_periods=1).sum()
    else:
        return shifted.rolling(n, min_periods=1).agg(agg)


def _grp_apply(df: pd.DataFrame, key_col: str, src_col: str,
               n: int, agg: str = "mean") -> pd.Series:
    """groupby + _rolling_shift をラップしてインデックスを整合させる"""
    return (
        df.groupby(key_col)[src_col]
        .apply(lambda x: _rolling_shift(x, n, agg))
        .reset_index(level=0, drop=True)
    )


def _make_speed_index(df: pd.DataFrame) -> pd.Series:
    """スピード指数（簡易版）: 距離 / タイム × 補正係数"""
    time_col  = _detect_col(df, ["time", "tymei", "record_time", "finish_time"])
    kyori_col = _detect_col(df, ["kyori", "distance"])
    if time_col is None or kyori_col is None:
        return pd.Series(np.nan, index=df.index, name="_speed_index")

    time_sec = pd.to_numeric(df[time_col], errors="coerce")
    kyori    = pd.to_numeric(df[kyori_col], errors="coerce")

    # JRAのタイム記録は "秒" 単位が多いが "0秒台" の場合は 1/10 秒単位のことも
    # 1600m で 96 秒前後が標準 → そのまま割り算でOK
    speed = kyori / time_sec.replace(0, np.nan)
    return speed.rename("_speed_index")


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    時系列リークなし特徴量を生成する。
    内部作業列は '_' プレフィックスを付ける（後で削除）。
    """
    print(f"\n[STEP 3-4] 特徴量エンジニアリング")

    df = df.copy()

    # ─── 4-0. スピード指数（当レース結果 → 次レース以降で使用）───
    df["_speed_index"] = _make_speed_index(df)

    # ─── 4-1. 馬の過去実績 ───
    if "horse_id" in df.columns:
        print("  馬の過去実績を計算中...")
        grp_horse = df.groupby("horse_id")

        # 直近着順（1/3/5走平均）
        df["horse_jyuni_last1"]     = grp_horse["_jyuni"].apply(lambda x: x.shift(1)).reset_index(level=0, drop=True)
        df["horse_jyuni_mean3"]     = _grp_apply(df, "horse_id", "_jyuni",  3, "mean")
        df["horse_jyuni_mean5"]     = _grp_apply(df, "horse_id", "_jyuni",  5, "mean")
        df["horse_jyuni_mean10"]    = _grp_apply(df, "horse_id", "_jyuni", 10, "mean")

        # 勝率・複勝率（過去5・10走）
        df["horse_win_rate5"]       = _grp_apply(df, "horse_id", TARGET_WIN,   5, "mean")
        df["horse_win_rate10"]      = _grp_apply(df, "horse_id", TARGET_WIN,  10, "mean")
        df["horse_place_rate5"]     = _grp_apply(df, "horse_id", TARGET_PLACE, 5, "mean")
        df["horse_place_rate10"]    = _grp_apply(df, "horse_id", TARGET_PLACE,10, "mean")

        # スピード指数（過去）
        df["horse_speed_last1"]     = grp_horse["_speed_index"].apply(lambda x: x.shift(1)).reset_index(level=0, drop=True)
        df["horse_speed_mean3"]     = _grp_apply(df, "horse_id", "_speed_index", 3, "mean")
        df["horse_speed_mean5"]     = _grp_apply(df, "horse_id", "_speed_index", 5, "mean")
        df["horse_speed_max5"]      = _grp_apply(df, "horse_id", "_speed_index", 5, "max")
        df["horse_speed_std5"]      = _grp_apply(df, "horse_id", "_speed_index", 5, "std")

        # 通算出走数（過去のみ）
        df["horse_total_runs"]      = grp_horse.cumcount()

        # 前走からの休養日数
        df["_race_date_days"] = (df["race_date"] - pd.Timestamp("2000-01-01")).dt.days
        df["days_since_last"] = (
            df.groupby("horse_id")["_race_date_days"]
            .apply(lambda x: x - x.shift(1))
            .reset_index(level=0, drop=True)
        )
        # 休養カテゴリ（連闘/中1週以内/通常/長期休み）
        df["rest_category"] = pd.cut(
            df["days_since_last"],
            bins=[-1, 7, 14, 35, 90, 9999],
            labels=[0, 1, 2, 3, 4]
        ).astype("float32")

        # 人気薄激走歴（過去5走で5番人気以下かつ3着以内の頻度）
        if "_ninki_raw" in df.columns:
            df["_was_upset"] = (
                (df["_ninki_raw"] >= 5) & (df[TARGET_PLACE] == 1)
            ).astype(np.int8)
            df["horse_upset_rate5"] = _grp_apply(df, "horse_id", "_was_upset", 5, "mean")

    # ─── 4-2. 騎手の過去実績 ───
    if "jockey_id" in df.columns:
        print("  騎手の過去実績を計算中...")
        df["jockey_win_rate20"]   = _grp_apply(df, "jockey_id", TARGET_WIN,   20, "mean")
        df["jockey_place_rate20"] = _grp_apply(df, "jockey_id", TARGET_PLACE, 20, "mean")
        df["jockey_win_rate50"]   = _grp_apply(df, "jockey_id", TARGET_WIN,   50, "mean")

    # ─── 4-3. 調教師の過去実績 ───
    if "trainer_id" in df.columns:
        print("  調教師の過去実績を計算中...")
        df["trainer_win_rate20"]   = _grp_apply(df, "trainer_id", TARGET_WIN,   20, "mean")
        df["trainer_place_rate20"] = _grp_apply(df, "trainer_id", TARGET_PLACE, 20, "mean")

    # ─── 4-4. 馬×コース/距離別の成績 ───
    if "horse_id" in df.columns:
        print("  馬×コース成績を計算中...")

        for course_key in ["jyo_cd", "kyori"]:
            if course_key in df.columns:
                grp_key = f"horse_id__{course_key}"
                df[grp_key] = df["horse_id"].astype(str) + "_" + df[course_key].astype(str)
                col_name = f"horse_win_rate5_{course_key}"
                df[col_name] = _grp_apply(df, grp_key, TARGET_WIN, 5, "mean")
                df.drop(columns=[grp_key], inplace=True)

    # ─── 4-5. 馬×騎手コンビの成績 ───
    if "horse_id" in df.columns and "jockey_id" in df.columns:
        combo_key = df["horse_id"].astype(str) + "_" + df["jockey_id"].astype(str)
        df["_horse_jockey_key"] = combo_key
        df["combo_win_rate10"]   = _grp_apply(df, "_horse_jockey_key", TARGET_WIN,   10, "mean")
        df["combo_place_rate10"] = _grp_apply(df, "_horse_jockey_key", TARGET_PLACE, 10, "mean")
        df.drop(columns=["_horse_jockey_key"], inplace=True)

    # ─── 4-6. フィールドレベル特徴量 ───
    print("  フィールド特徴量を計算中...")

    # 枠番バイアス（内/外/中）
    if "waku_no" in df.columns:
        waku = pd.to_numeric(df["waku_no"], errors="coerce")
        df["waku_inner_flag"] = (waku <= 2).astype(np.int8)
        df["waku_outer_flag"] = (waku >= 7).astype(np.int8)
        df["waku_bias"] = (waku - 1) / df["field_size"].clip(lower=1)

    # 距離カテゴリ
    kyori_col = _detect_col(df, ["kyori", "distance"])
    if kyori_col:
        kyori = pd.to_numeric(df[kyori_col], errors="coerce")
        df["kyori_num"] = kyori
        df["dist_cat"] = pd.cut(
            kyori,
            bins=[0, 1200, 1500, 1800, 2000, 2400, 9999],
            labels=[0, 1, 2, 3, 4, 5]
        ).astype("float32")

    # ─── 4-7. 同レース内でのスコア相対化（leak-freeに注意）───
    # 同レース内では未来の情報は含まないため、同レース内の過去馬のデータを集計するのは
    # 順序依存になる。ここでは「直近速度の同レース平均」のみ使用
    if "horse_speed_mean5" in df.columns:
        race_speed_mean = df.groupby("race_id")["horse_speed_mean5"].transform("mean")
        df["rel_speed_vs_field"] = df["horse_speed_mean5"] - race_speed_mean

    # ─── 4-8. 荒れ度フィルター用特徴量 ───
    # 過去同コース・同クラスの払戻平均（バックテスト時の荒れレース判定に使用）
    # ここではフラグを作るのではなく、過去実績の代理指標を作る
    if "_tansho_payout_raw" in df.columns and "jyo_cd" in df.columns:
        df["_past_payout_tmp"] = df["_tansho_payout_raw"]
        df["venue_avg_payout5"] = _grp_apply(
            df, "jyo_cd", "_past_payout_tmp", 5, "mean"
        )

    # ─── 4-9. 連続出走パターン ───
    if "horse_id" in df.columns:
        # 前走着順と今走距離差（前走の距離との差）
        kyori_col2 = _detect_col(df, ["kyori_num", "kyori", "distance"])
        if kyori_col2 and kyori_col2 in df.columns:
            df["dist_change"] = (
                df.groupby("horse_id")[kyori_col2]
                .apply(lambda x: x - x.shift(1))
                .reset_index(level=0, drop=True)
            )

    # ─── 4-10. 馬齢数値化 ───
    age_col = _detect_col(df, ["age", "barei"])
    if age_col:
        df["age_num"] = pd.to_numeric(df[age_col], errors="coerce")

    feat_cols = [c for c in df.columns if not c.startswith("_")]
    print(f"  特徴量生成完了: {len(feat_cols)} 列")
    return df


# ═══════════════════════════════════════════════
# ステップ 5: カテゴリ変換・ブラックリスト削除
# ═══════════════════════════════════════════════

def finalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. カテゴリ変数を category 型に変換（LightGBM ネイティブ処理用）
    2. ブラックリスト列を強制削除（安全装置）
    3. 内部作業用 '_' 列を削除（ターゲット・pay列は保持）
    """
    print(f"\n[STEP 3-5] 特徴量最終化 (カテゴリ変換 + ブラックリスト削除)")

    # ─── カテゴリ変換 ───
    converted = []
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
            converted.append(col)
    print(f"  category型に変換: {converted}")

    # ─── ブラックリスト強制削除 ───
    to_drop = [c for c in FORBIDDEN_COLS if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)
        print(f"  ブラックリスト削除: {to_drop}")
    else:
        print("  ブラックリスト対象列なし（すでに安全）")

    # ─── 内部作業列の削除（'_' プレフィックス）───
    # ただし '_jyuni', '_tansho_payout_raw', '_fukusho_payout_raw',
    #         '_odds_tansho_raw', '_ninki_raw' はバックテスト用に保持
    keep_internal = {
        "_tansho_payout_raw", "_fukusho_payout_raw",
        "_odds_tansho_raw",   "_ninki_raw",
    }
    drop_internal = [
        c for c in df.columns
        if c.startswith("_") and c not in keep_internal
    ]
    df = df.drop(columns=drop_internal)

    print(f"  最終列数: {len(df.columns)}")
    return df


# ═══════════════════════════════════════════════
# メイン実行
# ═══════════════════════════════════════════════

def main() -> pd.DataFrame:
    print("\n" + "="*60)
    print(" VER8 STEP 3: 学習データ作成")
    print("="*60)

    # 1. 読み込み
    df = load_from_db(DB_PATH)

    # 2. 前処理
    df = basic_preprocess(df)

    # 3. ターゲット作成
    df = create_targets(df)

    # 4. 特徴量生成
    df = create_features(df)

    # 5. 最終化
    df = finalize(df)

    # 6. 保存
    out_path = DATA_DIR / "training_data.parquet"
    df.to_parquet(out_path, index=False)

    # サマリー
    print(f"\n{'='*60}")
    print(f"✅ 保存完了: {out_path}")
    print(f"   形状     : {df.shape}")
    print(f"   期間     : {df['race_date'].min().date()} 〜 {df['race_date'].max().date()}")
    print(f"   勝率     : {df[TARGET_WIN].mean():.3f}")
    print(f"   複勝率   : {df[TARGET_PLACE].mean():.3f}")
    print(f"   nullチェック (上位10列):")
    null_rates = df.isnull().mean().sort_values(ascending=False).head(10)
    for col, rate in null_rates.items():
        print(f"     {col:<40} {rate:.1%}")
    print("="*60)

    return df


if __name__ == "__main__":
    df = main()
