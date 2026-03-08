"""
step5_backtest.py - ver8
=========================
多様な馬券種に対応した網羅的バックテスト + ROI算出

【馬券戦略】
1. 単勝（Tansho）    : EV閾値 + 最低勝率でフィルタ
2. 複勝（Fukusho）   : EV閾値でフィルタ
3. 馬連ボックス       : EV上位N頭のボックス（2頭以上のレースのみ）
4. 三連複ボックス     : EV上位N頭のボックス（3頭以上のレースのみ）
5. 馬連軸流し（流し） : EV最上位を軸に残りへ流し

【荒れ度フィルター】
- 過去の払戻傾向からレースの「荒れやすさ」を推定
- 荒れやすいレースでは高配当狙い（馬連/三連複）を優先

実行方法:
    python step5_backtest.py
"""

import sys
import json
import pickle
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    MODEL_DIR, OUTPUT_DIR,
    BET_UNIT, MIN_WIN_PROB, MIN_EV_TANSHO, MIN_EV_PLACE,
    BOX_TOP_N, NAGASHI_PIVOT_N, ARERI_PAYOUT_LINE,
    TARGET_WIN, TARGET_PLACE,
)

# ═══════════════════════════════════════════════
# データ読み込み
# ═══════════════════════════════════════════════

def load_predictions() -> pd.DataFrame:
    """step4 で保存したテスト予測を読み込む"""
    pred_path = MODEL_DIR / "test_predictions.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"{pred_path} が見つかりません。先に step4 を実行してください。"
        )
    df = pd.read_parquet(pred_path)
    df["race_date"] = pd.to_datetime(df["race_date"])
    df = df.sort_values(["race_date", "race_id", "uma_no"]).reset_index(drop=True)
    print(f"予測データ読み込み: {len(df):,} 行")
    print(f"  期間: {df['race_date'].min().date()} 〜 {df['race_date'].max().date()}")
    return df


# ═══════════════════════════════════════════════
# 期待値（EV）計算
# ═══════════════════════════════════════════════

def compute_ev(df: pd.DataFrame) -> pd.DataFrame:
    """
    各馬の期待値を計算する。
    EV = 予測勝率 × オッズ - 1
    オッズが利用できない場合は払戻履歴から推定しない（NaNのまま）。
    """
    df = df.copy()

    # 単勝EV = pred_win_cal × (odds / 100) - 1
    # ※ 日本のオッズは "何倍" 表記ではなく "払戻金額(100円ベット時)" が多い
    #   例: オッズ5.0倍 → payout = 500円
    if "_odds_tansho_raw" in df.columns:
        odds = df["_odds_tansho_raw"].clip(lower=1.0)
        df["ev_win"] = df["pred_win_cal"] * odds - 1.0
    elif "_tansho_payout_raw" in df.columns:
        # 払戻=オッズ×100 で換算
        pay = df["_tansho_payout_raw"].clip(lower=100.0)
        df["ev_win"] = df["pred_win_cal"] * (pay / 100.0) - 1.0
    else:
        df["ev_win"] = np.nan

    # 複勝EV（払戻があれば）
    if "_fukusho_payout_raw" in df.columns:
        pay_fuku = df["_fukusho_payout_raw"].clip(lower=100.0)
        df["ev_place"] = df["pred_place_cal"] * (pay_fuku / 100.0) - 1.0
    else:
        df["ev_place"] = np.nan

    return df


# ═══════════════════════════════════════════════
# 単勝バックテスト
# ═══════════════════════════════════════════════

def backtest_tansho(df: pd.DataFrame) -> pd.DataFrame:
    """
    単勝バックテスト。
    買い条件: ev_win > MIN_EV_TANSHO AND pred_win_cal > MIN_WIN_PROB
    """
    cond = (
        df["ev_win"].notna() &
        (df["ev_win"] > MIN_EV_TANSHO) &
        (df["pred_win_cal"] > MIN_WIN_PROB)
    )
    bets = df[cond].copy()

    if len(bets) == 0:
        print("  ⚠ 単勝: 買い対象なし")
        return pd.DataFrame()

    bets["bet_cost"]   = BET_UNIT
    bets["bet_return"] = np.where(
        bets[TARGET_WIN] == 1,
        bets["_tansho_payout_raw"].fillna(0),
        0.0
    )
    bets["profit"] = bets["bet_return"] - bets["bet_cost"]
    bets["strategy"] = "tansho"

    return bets[["race_date", "race_id", "uma_no",
                 "pred_win_cal", "ev_win", "_odds_tansho_raw",
                 TARGET_WIN, "bet_cost", "bet_return", "profit", "strategy"]]


# ═══════════════════════════════════════════════
# 複勝バックテスト
# ═══════════════════════════════════════════════

def backtest_fukusho(df: pd.DataFrame) -> pd.DataFrame:
    """複勝バックテスト"""
    if "ev_place" not in df.columns or df["ev_place"].isna().all():
        return pd.DataFrame()

    cond = (
        df["ev_place"].notna() &
        (df["ev_place"] > MIN_EV_PLACE) &
        (df["pred_place_cal"] > 0.35)
    )
    bets = df[cond].copy()

    if len(bets) == 0:
        return pd.DataFrame()

    bets["bet_cost"]   = BET_UNIT
    bets["bet_return"] = np.where(
        bets[TARGET_PLACE] == 1,
        bets["_fukusho_payout_raw"].fillna(0),
        0.0
    )
    bets["profit"] = bets["bet_return"] - bets["bet_cost"]
    bets["strategy"] = "fukusho"

    return bets[["race_date", "race_id", "uma_no",
                 "pred_place_cal", "ev_place",
                 TARGET_PLACE, "bet_cost", "bet_return", "profit", "strategy"]]


# ═══════════════════════════════════════════════
# 馬連ボックス バックテスト
# ═══════════════════════════════════════════════

def backtest_umaren_box(df: pd.DataFrame) -> pd.DataFrame:
    """
    EV上位 BOX_TOP_N 頭の馬連ボックス。
    実際の馬連払戻がない場合は結果確認のみ（ROI計算なし）。
    """
    results = []
    has_umaren = "_umaren_payout" in df.columns or "payout_umaren" in df.columns

    payout_umaren_col = next(
        (c for c in ["_umaren_payout", "payout_umaren"] if c in df.columns), None
    )

    for race_id, race_df in df.groupby("race_id"):
        if len(race_df) < 2:
            continue

        # EV上位N頭を選択（ev_win で順位付け）
        top = race_df.nlargest(BOX_TOP_N, "ev_win", keep="all")
        if len(top) < 2:
            continue

        # ボックス組み合わせ
        combos = list(combinations(top["uma_no"].tolist(), 2))
        n_bets = len(combos)
        total_cost = BET_UNIT * n_bets

        # 実際の1着・2着馬番
        win_horses  = set(race_df[race_df[TARGET_WIN] == 1]["uma_no"].tolist())
        place_horses = set(race_df[race_df["_ninki_raw"].notna()]["uma_no"].tolist()
                           if "_ninki_raw" in race_df.columns else [])
        # 2着（kakutei_jyuni == 2）
        jyuni2_col = None  # 着順列はFORBIDDEN_COLSで削除済みなのでTargetから推測
        # TARGET_WIN=1着のみ確認できるためwinとplaceから2着推定
        # place(3着以内) & not win(1着でない) の中から最高EV
        place_not_win = race_df[(race_df[TARGET_PLACE] == 1) & (race_df[TARGET_WIN] == 0)]

        top_uma_set = set(top["uma_no"].tolist())

        # 馬連的中判定：1着と2着がともにボックス内にいるか
        win_uma    = race_df[race_df[TARGET_WIN] == 1]["uma_no"].tolist()
        # 2着の馬番は place & not win から1頭（EV最高）と仮定
        second_uma = place_not_win.sort_values("pred_win_cal", ascending=False)["uma_no"].tolist()[:1]

        hit = False
        if win_uma and second_uma:
            hit = (win_uma[0] in top_uma_set) and (second_uma[0] in top_uma_set)

        ret = 0.0
        if hit and payout_umaren_col:
            race_pay = race_df[payout_umaren_col].dropna()
            if len(race_pay) > 0:
                ret = race_pay.iloc[0]

        results.append({
            "race_date":  race_df["race_date"].iloc[0],
            "race_id":    race_id,
            "n_bets":     n_bets,
            "bet_cost":   total_cost,
            "bet_return": ret,
            "profit":     ret - total_cost,
            "hit":        int(hit),
            "strategy":   "umaren_box",
        })

    return pd.DataFrame(results) if results else pd.DataFrame()


# ═══════════════════════════════════════════════
# 三連複ボックス バックテスト
# ═══════════════════════════════════════════════

def backtest_sanrenpuku_box(df: pd.DataFrame) -> pd.DataFrame:
    """EV上位 BOX_TOP_N 頭の三連複ボックス"""
    results = []
    payout_col = next(
        (c for c in ["_sanrenpuku_payout", "payout_sanrenpuku"] if c in df.columns), None
    )

    for race_id, race_df in df.groupby("race_id"):
        if len(race_df) < 3:
            continue

        top = race_df.nlargest(BOX_TOP_N, "ev_win", keep="all")
        if len(top) < 3:
            continue

        combos    = list(combinations(top["uma_no"].tolist(), 3))
        n_bets    = len(combos)
        total_cost = BET_UNIT * n_bets

        # 3着以内馬番を取得
        top3_horses = set(race_df[race_df[TARGET_PLACE] == 1]["uma_no"].tolist())
        top_uma_set = set(top["uma_no"].tolist())

        # 三連複的中：3着以内3頭がすべてボックス内に入っているか
        hit = len(top3_horses.intersection(top_uma_set)) >= 3

        ret = 0.0
        if hit and payout_col:
            race_pay = race_df[payout_col].dropna()
            if len(race_pay) > 0:
                ret = race_pay.iloc[0]

        results.append({
            "race_date":  race_df["race_date"].iloc[0],
            "race_id":    race_id,
            "n_bets":     n_bets,
            "bet_cost":   total_cost,
            "bet_return": ret,
            "profit":     ret - total_cost,
            "hit":        int(hit),
            "strategy":   "sanrenpuku_box",
        })

    return pd.DataFrame(results) if results else pd.DataFrame()


# ═══════════════════════════════════════════════
# 馬連 軸流し バックテスト
# ═══════════════════════════════════════════════

def backtest_umaren_nagashi(df: pd.DataFrame) -> pd.DataFrame:
    """EV最上位を軸に、EV上位N頭へ流す馬連軸流し"""
    results = []
    payout_col = next(
        (c for c in ["_umaren_payout", "payout_umaren"] if c in df.columns), None
    )

    for race_id, race_df in df.groupby("race_id"):
        if len(race_df) < 3:
            continue

        sorted_df = race_df.sort_values("ev_win", ascending=False).reset_index(drop=True)
        pivot     = sorted_df.iloc[0]          # EV最大の軸馬
        aite      = sorted_df.iloc[1:BOX_TOP_N] # 相手馬

        if len(aite) == 0:
            continue

        n_bets     = len(aite)
        total_cost = BET_UNIT * n_bets

        pivot_uma = pivot["uma_no"]
        aite_uma  = set(aite["uma_no"].tolist())

        # 2着馬番推定（place & not win から）
        win_uma = race_df[race_df[TARGET_WIN] == 1]["uma_no"].tolist()
        second_candidates = race_df[
            (race_df[TARGET_PLACE] == 1) & (race_df[TARGET_WIN] == 0)
        ].sort_values("pred_win_cal", ascending=False)["uma_no"].tolist()
        second_uma = second_candidates[:1]

        hit = False
        if win_uma and second_uma:
            result_set = {win_uma[0], second_uma[0]}
            hit = (pivot_uma in result_set) and bool(result_set - {pivot_uma} & aite_uma)

        ret = 0.0
        if hit and payout_col:
            race_pay = race_df[payout_col].dropna()
            if len(race_pay) > 0:
                ret = race_pay.iloc[0]

        results.append({
            "race_date":  race_df["race_date"].iloc[0],
            "race_id":    race_id,
            "pivot_uma":  pivot_uma,
            "n_bets":     n_bets,
            "bet_cost":   total_cost,
            "bet_return": ret,
            "profit":     ret - total_cost,
            "hit":        int(hit),
            "strategy":   "umaren_nagashi",
        })

    return pd.DataFrame(results) if results else pd.DataFrame()


# ═══════════════════════════════════════════════
# 荒れ度フィルター付き戦略
# ═══════════════════════════════════════════════

def apply_areri_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    レースを「荒れやすい」「堅い」に二分する。
    venue_avg_payout5 が ARERI_PAYOUT_LINE を超えるレースを荒れレースとする。
    """
    if "venue_avg_payout5" not in df.columns:
        return df, pd.DataFrame()

    areri_mask = df.groupby("race_id")["venue_avg_payout5"].transform("mean") > ARERI_PAYOUT_LINE
    df_areri   = df[areri_mask].copy()
    df_solid   = df[~areri_mask].copy()

    return df_solid, df_areri


# ═══════════════════════════════════════════════
# サマリーレポート生成
# ═══════════════════════════════════════════════

def compute_roi(bets_df: pd.DataFrame, strategy_name: str) -> dict:
    """ROI・的中率・累積損益を計算してサマリーを返す"""
    if len(bets_df) == 0:
        return {}

    total_cost   = bets_df["bet_cost"].sum()
    total_return = bets_df["bet_return"].sum()
    total_profit = bets_df["profit"].sum()
    roi          = total_return / total_cost if total_cost > 0 else 0.0

    if "hit" in bets_df.columns:
        hit_rate = bets_df["hit"].mean()
        n_hit    = int(bets_df["hit"].sum())
    elif TARGET_WIN in bets_df.columns:
        hit_rate = bets_df[TARGET_WIN].mean()
        n_hit    = int(bets_df[TARGET_WIN].sum())
    elif TARGET_PLACE in bets_df.columns:
        hit_rate = bets_df[TARGET_PLACE].mean()
        n_hit    = int(bets_df[TARGET_PLACE].sum())
    else:
        hit_rate = 0.0
        n_hit    = 0

    n_races = bets_df.groupby("race_id").ngroups if "race_id" in bets_df.columns else len(bets_df)

    summary = {
        "strategy":      strategy_name,
        "n_bets":        len(bets_df),
        "n_races":       n_races,
        "n_hit":         n_hit,
        "hit_rate":      round(hit_rate, 4),
        "total_cost":    int(total_cost),
        "total_return":  int(total_return),
        "total_profit":  int(total_profit),
        "roi":           round(roi, 4),
        "avg_profit":    round(total_profit / len(bets_df), 1),
    }

    print(f"\n  [{strategy_name}]")
    print(f"    買い数   : {summary['n_bets']:,} 点 ({summary['n_races']:,} レース)")
    print(f"    的中率   : {summary['hit_rate']:.1%}  ({summary['n_hit']} 的中)")
    print(f"    投資額   : ¥{summary['total_cost']:,}")
    print(f"    回収額   : ¥{summary['total_return']:,}")
    print(f"    損益     : ¥{summary['total_profit']:,}")
    print(f"    ROI      : {summary['roi']:.2%}")

    return summary


# ═══════════════════════════════════════════════
# グラフ描画
# ═══════════════════════════════════════════════

def plot_cumulative_roi(bets_dict: dict[str, pd.DataFrame], title: str = "累積収支"):
    """各戦略の累積収支グラフを描画"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # ─── 累積損益 ───
    ax1 = axes[0]
    for name, bets in bets_dict.items():
        if len(bets) == 0 or "profit" not in bets.columns:
            continue
        bets_sorted = bets.sort_values("race_date")
        cum_profit  = bets_sorted["profit"].cumsum()
        ax1.plot(
            bets_sorted["race_date"],
            cum_profit,
            label=name,
            linewidth=1.5,
        )
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.set_title(f"{title} - 累積損益（円）")
    ax1.set_xlabel("日付")
    ax1.set_ylabel("累積損益（円）")
    ax1.legend()
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"¥{x:,.0f}"))
    ax1.grid(True, alpha=0.3)

    # ─── 月次ROI ───
    ax2 = axes[1]
    for name, bets in bets_dict.items():
        if len(bets) == 0 or "profit" not in bets.columns:
            continue
        bets_sorted = bets.sort_values("race_date").copy()
        bets_sorted["ym"] = bets_sorted["race_date"].dt.to_period("M")
        monthly = bets_sorted.groupby("ym").agg(
            cost=("bet_cost", "sum"),
            ret=("bet_return", "sum"),
        )
        monthly["roi"] = monthly["ret"] / monthly["cost"].clip(lower=1)
        ax2.plot(
            monthly.index.to_timestamp(),
            monthly["roi"],
            label=name,
            marker="o",
            markersize=3,
            linewidth=1.2,
        )
    ax2.axhline(1.0, color="red", linewidth=1.0, linestyle="--", label="回収率 100%")
    ax2.axhline(0.75, color="gray", linewidth=0.8, linestyle=":", label="JRA控除率(参考)")
    ax2.set_title("月次 ROI")
    ax2.set_xlabel("月")
    ax2.set_ylabel("ROI")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "backtest_result.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  グラフ保存: {out_path}")


def plot_roi_by_period(bets_dict: dict[str, pd.DataFrame]):
    """月次ROIの時系列トレンドを確認（右肩下がりがないかチェック）"""
    if not bets_dict:
        return

    fig, ax = plt.subplots(figsize=(14, 5))

    for name, bets in bets_dict.items():
        if len(bets) == 0 or "profit" not in bets.columns:
            continue
        bets = bets.sort_values("race_date").copy()
        bets["ym"] = bets["race_date"].dt.to_period("M")
        monthly = bets.groupby("ym").agg(
            cost=("bet_cost", "sum"),
            ret=("bet_return", "sum"),
        )
        monthly["roi"] = monthly["ret"] / monthly["cost"].clip(lower=1)

        # 3ヶ月移動平均
        monthly["roi_ma3"] = monthly["roi"].rolling(3, min_periods=1).mean()

        ax.plot(
            monthly.index.to_timestamp(),
            monthly["roi_ma3"],
            label=f"{name} (3M MA)",
            linewidth=2,
        )

    ax.axhline(1.0, color="red", linewidth=1.0, linestyle="--")
    ax.set_title("月次ROI（3ヶ月移動平均） - 右肩下がりチェック")
    ax.set_ylabel("ROI")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "roi_trend.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ROIトレンドグラフ保存: {out_path}")


# ═══════════════════════════════════════════════
# メイン実行
# ═══════════════════════════════════════════════

def main():
    print("\n" + "="*60)
    print(" VER8 STEP 5: バックテスト")
    print("="*60)

    # ─── 予測データ読み込み ───
    df = load_predictions()

    # ─── EV計算 ───
    df = compute_ev(df)

    # ─── フィールド内でEVが全てNaNの場合は警告 ───
    ev_null_rate = df["ev_win"].isna().mean()
    if ev_null_rate > 0.5:
        print(f"⚠ ev_win のNull率が高い ({ev_null_rate:.1%})。"
              "オッズ列が利用できていない可能性があります。")

    # ─── 荒れ度フィルター ───
    df_solid, df_areri = apply_areri_filter(df)
    print(f"\n荒れ度フィルター:"
          f" 堅いレース {df_solid['race_id'].nunique():,}R"
          f" / 荒れレース {df_areri['race_id'].nunique():,}R")

    # ═══ 各戦略のバックテスト ═══
    print("\n" + "─"*60)
    print("バックテスト実行中...")
    print("─"*60)

    results_all = {}
    summaries   = []

    # 1. 全体: 単勝
    bets_tansho = backtest_tansho(df)
    results_all["単勝(全体)"] = bets_tansho
    if len(bets_tansho) > 0:
        summaries.append(compute_roi(bets_tansho, "単勝(全体)"))

    # 2. 全体: 複勝
    bets_fukusho = backtest_fukusho(df)
    results_all["複勝(全体)"] = bets_fukusho
    if len(bets_fukusho) > 0:
        summaries.append(compute_roi(bets_fukusho, "複勝(全体)"))

    # 3. 全体: 馬連ボックス
    bets_umaren_box = backtest_umaren_box(df)
    results_all["馬連BOX(全体)"] = bets_umaren_box
    if len(bets_umaren_box) > 0:
        summaries.append(compute_roi(bets_umaren_box, "馬連BOX(全体)"))

    # 4. 全体: 三連複ボックス
    bets_sanrenpuku = backtest_sanrenpuku_box(df)
    results_all["三連複BOX(全体)"] = bets_sanrenpuku
    if len(bets_sanrenpuku) > 0:
        summaries.append(compute_roi(bets_sanrenpuku, "三連複BOX(全体)"))

    # 5. 全体: 馬連軸流し
    bets_nagashi = backtest_umaren_nagashi(df)
    results_all["馬連流し(全体)"] = bets_nagashi
    if len(bets_nagashi) > 0:
        summaries.append(compute_roi(bets_nagashi, "馬連流し(全体)"))

    # 6. 荒れレース限定: 単勝
    if len(df_areri) > 0:
        bets_areri_tan = backtest_tansho(df_areri)
        results_all["単勝(荒れ)"] = bets_areri_tan
        if len(bets_areri_tan) > 0:
            summaries.append(compute_roi(bets_areri_tan, "単勝(荒れ)"))

    # 7. 堅いレース限定: 単勝
    if len(df_solid) > 0:
        bets_solid_tan = backtest_tansho(df_solid)
        results_all["単勝(堅い)"] = bets_solid_tan
        if len(bets_solid_tan) > 0:
            summaries.append(compute_roi(bets_solid_tan, "単勝(堅い)"))

    # ─── サマリーテーブル ───
    print("\n" + "="*60)
    print("バックテスト サマリー")
    print("="*60)
    if summaries:
        summary_df = pd.DataFrame(summaries)
        print(summary_df.to_string(index=False))

        # CSV保存
        summary_path = OUTPUT_DIR / "backtest_summary.csv"
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"\n  サマリー保存: {summary_path}")

    # ─── 詳細ベットログ保存 ───
    all_bets_list = []
    for strategy_name, bets in results_all.items():
        if len(bets) > 0:
            bets_copy = bets.copy()
            bets_copy["strategy_label"] = strategy_name
            all_bets_list.append(bets_copy)

    if all_bets_list:
        all_bets = pd.concat(all_bets_list, ignore_index=True)
        log_path = OUTPUT_DIR / "bet_log.parquet"
        all_bets.to_parquet(log_path, index=False)
        print(f"  ベットログ保存: {log_path}")

    # ─── グラフ描画 ───
    plot_cumulative_roi(
        {k: v for k, v in results_all.items() if len(v) > 0}
    )
    plot_roi_by_period(
        {k: v for k, v in results_all.items() if len(v) > 0}
    )

    # ─── 月別ROIテーブル（主戦略: 単勝全体）───
    if len(bets_tansho) > 0:
        print("\n" + "─"*60)
        print("単勝 月別ROI")
        print("─"*60)
        bt = bets_tansho.copy()
        bt["ym"] = bt["race_date"].dt.to_period("M")
        monthly = bt.groupby("ym").agg(
            n_bets=("bet_cost", "count"),
            cost=("bet_cost", "sum"),
            ret=("bet_return", "sum"),
            profit=("profit", "sum"),
        )
        monthly["roi"] = monthly["ret"] / monthly["cost"].clip(lower=1)
        monthly["roi_str"] = monthly["roi"].map(lambda x: f"{x:.1%}")
        print(monthly[["n_bets", "cost", "ret", "profit", "roi_str"]].to_string())

    print("\n" + "="*60)
    print("✅ STEP 5 完了")
    print("="*60)

    return results_all, summaries


if __name__ == "__main__":
    results_all, summaries = main()
