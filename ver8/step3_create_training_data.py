"""
================================================================================
競馬予測システム ver8 - Step 3: 学習データ作成
================================================================================

【ver7 からの根本的変更点】

1. Target Encoding 完全廃止
   - 騎手/調教師/種牡馬/母父 の勝率を手動で集計してエンコードするのを廃止
   - 代わりにこれらのコードを str/category 型のまま保持
   - LightGBM の categorical_feature に渡すことで数学的にリークを防ぐ

2. 週別成績キャッシュは過去累積のみ使用（カットオフ厳守）
   - 過去: jockey_win_rate, sire_win_rate などは「当該レース開催週の月曜前」まで
   - これにより同週の他レース結果がリークしない

3. FORBIDDEN_COLS 安全パージ（最終ステップ）
   - 特徴量生成の最後に FORBIDDEN_FEATURE_COLS を強制 drop
   - 絶対に当日情報が特徴量に残らないことを保証

4. 荒れ度ターゲット追加
   - target_turbulence = 1 if tansho_payout > 1000 else 0
   - 荒れ予測モデル用（step4で別途学習）

================================================================================
"""

import sqlite3
import pandas as pd
import numpy as np
import pickle
import os
import sys
import gc
import logging
import warnings
from datetime import datetime, timedelta
from scipy import stats as sp_stats

warnings.filterwarnings('ignore')

try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    sys.path.append(r'C:\Users\yhmor\JV\ver8')

from config import (Config, PathConfig, CodeMaster, TrainConfig,
                    FeatureConfig, TrainConfig, FORBIDDEN_FEATURE_COLS)


# =============================================================================
# ログ設定
# =============================================================================
def setup_logging():
    os.makedirs(PathConfig.LOG_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(f'{PathConfig.LOG_DIR}/step3_v8.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

logger = logging.getLogger(__name__)


# =============================================================================
# ユーティリティ関数
# =============================================================================
def parse_corner(val):
    if not val or str(val) in ('', '00', 'None'):
        return None
    try:
        s = str(val)
        return int(s.split('-')[0]) if '-' in s else int(s)
    except Exception:
        return None


def calc_ten_speed(time_sec, agari, kyori):
    if time_sec is None or agari is None or kyori is None:
        return None
    if kyori <= 600 or time_sec <= 0 or agari <= 0:
        return None
    zen_kyori = kyori - 600
    zen_time = time_sec - agari
    return zen_kyori / zen_time if zen_time > 0 else None


def classify_kyakushitsu(avg_corner4, avg_tosu):
    try:
        avg_corner4 = float(avg_corner4)
        avg_tosu    = float(avg_tosu)
    except (TypeError, ValueError):
        return 3
    if not avg_tosu or not avg_corner4:
        return 3
    ratio = avg_corner4 / avg_tosu
    if ratio <= 0.15:   return 1
    elif ratio <= 0.40: return 2
    elif ratio <= 0.70: return 3
    else:               return 4


def prev_week_monday(date_str):
    """レース開催日の「開催週月曜日」を返す（同週リーク防止）"""
    dt = datetime.strptime(str(date_str)[:8], '%Y%m%d')
    monday = dt - timedelta(days=dt.weekday())
    return monday.strftime('%Y%m%d')


def linear_slope(values):
    """線形回帰の傾きを返す（非欠損値が2点以上必要）"""
    vals = [v for v in values if v is not None and not pd.isna(v)]
    if len(vals) < 2:
        return 0.0
    x = np.arange(len(vals))
    slope, *_ = np.polyfit(x, vals, 1)
    return float(slope)


# =============================================================================
# DB接続クラス
# =============================================================================
class DBConnector:
    def __init__(self):
        self.conn = sqlite3.connect(PathConfig.DB_PATH, timeout=60)
        self.conn.row_factory = sqlite3.Row

    def close(self):
        self.conn.close()

    def get_tables(self):
        cur = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return [r[0] for r in cur.fetchall()]

    def get_columns(self, table):
        cur = self.conn.execute(f"PRAGMA table_info({table})")
        return [r[1] for r in cur.fetchall()]


# =============================================================================
# データ読み込みクラス
# =============================================================================
class RaceDataLoader:
    """N_RACE / N_UMA_RACE / N_HARAI から基本データを読み込む"""

    def __init__(self, db: DBConnector):
        self.db = db

    def load_base(self, start_year=2016, end_year=2026):
        """
        メインクエリ:
          N_RACE × N_UMA_RACE を JOIN してレース×馬単位のデータを取得。
          着順・払戻はターゲット計算にのみ使用し、
          最終的に特徴量からは drop する。
        """
        logger.info(f"ベースデータ読み込み: {start_year}〜{end_year}")

        # ── カラム存在チェック ─────────────────────────────────────────────
        race_cols  = self.db.get_columns('N_RACE')
        uma_cols   = self.db.get_columns('N_UMA_RACE')

        def opt(tbl, col, alias=None):
            """列が存在すればSELECT、なければ NULL"""
            alias = alias or col
            cols  = race_cols if tbl == 'r' else uma_cols
            return f"{tbl}.{col} AS {alias}" if col in cols else f"NULL AS {alias}"

        # ──────────────────────────────────────────────────────────────────
        # 正しいカラム名（PRAGMA table_info で確認済み）
        #   N_RACE      : SyussoTosu（×SyutsubaTosu）
        #   N_UMA_RACE  : BaTaijyu（×Bataiju）, Odds（×OddsTansho）
        #                 Ninki（×TanshoNinki）, Jyuni1c〜4c（×Corner1〜4）
        #                 HaronTimeL3（×AgariTime）, KettoNum1=父, KettoNum3=母父
        #                 KyakusituKubun（脚質区分）
        # ──────────────────────────────────────────────────────────────────
        sql = f"""
        SELECT
            r.Year || r.MonthDay || r.JyoCD || printf('%02d', r.RaceNum) AS race_id,
            r.Year || r.MonthDay AS race_date,
            r.Year AS year,
            r.JyoCD,
            r.RaceNum,
            r.TrackCD,
            r.Kyori AS kyori,
            r.SyussoTosu AS syusso_tosu,
            r.GradeCD,
            r.TenkoCD,
            r.SibaBabaCD,
            r.DirtBabaCD,
            ur.KettoNum,
            ur.Umaban AS umaban,
            ur.Wakuban AS wakuban,
            ur.KisyuCode,
            ur.ChokyosiCode,
            ur.KettoNum1 AS SireID,
            ur.KettoNum3 AS BmsID,
            ur.Barei AS barei,
            ur.SexCD,
            ur.Futan AS futan,
            ur.KakuteiJyuni AS kakutei_jyuni,
            ur.BaTaijyu AS bataiju,
            ur.ZogenSa AS zogen_sa,
            ur.ZogenFugo AS zogen_cd,
            CAST(ur.Odds AS REAL) / 10.0 AS odds_tansho,
            ur.Ninki AS tansho_ninki,
            ur.Jyuni1c AS corner1_raw,
            ur.Jyuni2c AS corner2_raw,
            ur.Jyuni3c AS corner3_raw,
            ur.Jyuni4c AS corner4_raw,
            ur.Time AS time_raw,
            CAST(ur.HaronTimeL3 AS REAL) / 10.0 AS agari_raw,
            ur.KyakusituKubun AS kyakushitsu_cd,
            ur.MinaraiCD AS minarai_cd,
            NULL AS mining_time_idx,
            NULL AS mining_match_idx,
            {opt('ur', 'Blinker', 'blinker_cd')}
        FROM N_RACE r
        INNER JOIN N_UMA_RACE ur
            ON r.Year = ur.Year AND r.MonthDay = ur.MonthDay
            AND r.JyoCD = ur.JyoCD AND r.RaceNum = ur.RaceNum
        WHERE r.Year >= '{start_year}' AND r.Year <= '{end_year}'
        ORDER BY r.Year, r.MonthDay, r.JyoCD, r.RaceNum, ur.Umaban
        """
        df = pd.read_sql_query(sql, self.db.conn)
        logger.info(f"  読み込み: {len(df):,} 行")
        return df

    def load_payouts(self, start_year=2016, end_year=2026):
        """
        N_HARAI から単勝・複勝払戻を読み込む（ターゲット計算用）

        N_HARAI の正確なカラム名（PRAGMA table_info確認済み）:
          PayTansyoUmaban1  : 単勝1着馬の馬番
          PayTansyoPay1     : 単勝払戻金（円, 例: 000000860 = 860円）
          PayFukusyoUmaban1〜5 : 複勝各着馬の馬番
          PayFukusyoPay1〜5    : 複勝払戻金（各馬）
          PayUmarenKumi1〜3, PayUmarenPay1〜3  : 馬連
          PayWideKumi1〜7,  PayWidePay1〜7     : ワイド
          PaySanrenpukuKumi1〜3, PaySanrenpukuPay1〜3 : 三連複
        """
        logger.info("払戻データ読み込み中...")

        # ── 単勝払戻（勝ち馬のみ1点）────────────────────────────────────
        # ── 複勝払戻（3着以内各馬）────────────────────────────────────────
        # race_id × horse_umaban レベルに展開し、df_base と馬番で結合する
        sql = f"""
        SELECT
            Year || MonthDay || JyoCD || printf('%02d', RaceNum) AS race_id,
            CAST(PayTansyoPay1 AS REAL) AS tansho_pay1,
            PayTansyoUmaban1 AS tansho_umaban1,
            CAST(PayFukusyoPay1 AS REAL) AS fukusho_pay1,
            PayFukusyoUmaban1 AS fukusho_umaban1,
            CAST(PayFukusyoPay2 AS REAL) AS fukusho_pay2,
            PayFukusyoUmaban2 AS fukusho_umaban2,
            CAST(PayFukusyoPay3 AS REAL) AS fukusho_pay3,
            PayFukusyoUmaban3 AS fukusho_umaban3,
            CAST(PayFukusyoPay4 AS REAL) AS fukusho_pay4,
            PayFukusyoUmaban4 AS fukusho_umaban4,
            CAST(PayFukusyoPay5 AS REAL) AS fukusho_pay5,
            PayFukusyoUmaban5 AS fukusho_umaban5
        FROM N_HARAI
        WHERE Year >= '{start_year}' AND Year <= '{end_year}'
        """
        try:
            df = pd.read_sql_query(sql, self.db.conn)

            # ── race_id × umaban レベルに展開 ──────────────────────────────
            rows = []
            for _, r in df.iterrows():
                race_id = r['race_id']
                # 単勝: 1着馬のみ
                try:
                    tansho_ub  = int(str(r['tansho_umaban1']).strip())
                    tansho_pay = float(r['tansho_pay1']) if r['tansho_pay1'] else 0
                    if tansho_ub > 0:
                        rows.append({
                            'race_id':        race_id,
                            'umaban':         tansho_ub,
                            'tansho_payout':  tansho_pay,
                            'fukusho_payout': 0,
                        })
                except Exception:
                    pass
                # 複勝: 3着以内各馬
                for k in range(1, 6):
                    try:
                        fu_ub  = int(str(r.get(f'fukusho_umaban{k}', 0) or 0).strip())
                        fu_pay = float(r.get(f'fukusho_pay{k}', 0) or 0)
                        if fu_ub > 0 and fu_pay > 0:
                            # 既に登録済み（単勝と同じ馬番）なら fukusho_payout を上書き
                            existing = [x for x in rows
                                        if x['race_id'] == race_id and x['umaban'] == fu_ub]
                            if existing:
                                existing[0]['fukusho_payout'] = fu_pay
                            else:
                                rows.append({
                                    'race_id':        race_id,
                                    'umaban':         fu_ub,
                                    'tansho_payout':  0,
                                    'fukusho_payout': fu_pay,
                                })
                    except Exception:
                        pass

            df_pay = pd.DataFrame(rows) if rows else pd.DataFrame(
                columns=['race_id', 'umaban', 'tansho_payout', 'fukusho_payout'])
            logger.info(f"  払戻データ: {len(df):,} レース → {len(df_pay):,} 行(馬別)")
            return df_pay

        except Exception as e:
            logger.warning(f"  払戻データ読み込みエラー: {e}")
            return pd.DataFrame()

    def load_kaisai(self, start_year=2016, end_year=2026):
        """N_KAISAI から天候・馬場状態・バイアス情報を読み込む"""
        logger.info("開催データ読み込み中...")
        try:
            kaisai_cols = self.db.get_columns('N_KAISAI')

            def opt_kaisai(col, alias=None):
                alias = alias or col
                return f"{col} AS {alias}" if col in kaisai_cols else f"NULL AS {alias}"

            sql = f"""
            SELECT
                Year || MonthDay || JyoCD AS kaisai_key,
                {opt_kaisai('TenkoCD')},
                {opt_kaisai('SibaBabaCD')},
                {opt_kaisai('DirtBabaCD')},
                {opt_kaisai('TempBiasInner', 'track_bias_inner')},
                {opt_kaisai('TempBiasPace', 'track_bias_pace')}
            FROM N_KAISAI
            WHERE Year >= '{start_year}' AND Year <= '{end_year}'
            """
            df = pd.read_sql_query(sql, self.db.conn)
            logger.info(f"  開催データ: {len(df):,} 行")
            return df
        except Exception as e:
            logger.warning(f"  開催データ読み込みエラー: {e}")
            return pd.DataFrame()

    def load_training_data(self, start_year=2016, end_year=2026):
        """調教データ（N_CHOKYOU テーブルがある場合）"""
        logger.info("調教データ読み込み中...")
        try:
            tables = self.db.get_tables()
            if 'N_CHOKYOU' not in tables and 'N_TRAINING' not in tables:
                logger.info("  調教テーブルなし → スキップ")
                return pd.DataFrame()

            tbl = 'N_CHOKYOU' if 'N_CHOKYOU' in tables else 'N_TRAINING'
            sql = f"""
            SELECT
                Year || MonthDay || JyoCD || printf('%02d', RaceNum) AS race_id,
                KettoNum,
                COUNT(*) AS total_train_cnt,
                SUM(CASE WHEN CourseCD='A' THEN 1 ELSE 0 END) AS hanro_count,
                SUM(CASE WHEN CourseCD='W' THEN 1 ELSE 0 END) AS wood_count
            FROM {tbl}
            WHERE Year >= '{start_year}' AND Year <= '{end_year}'
            GROUP BY race_id, KettoNum
            """
            df = pd.read_sql_query(sql, self.db.conn)
            logger.info(f"  調教データ: {len(df):,} 行")
            return df
        except Exception as e:
            logger.warning(f"  調教データ読み込みエラー: {e}")
            return pd.DataFrame()


# =============================================================================
# 過去成績計算クラス
# =============================================================================
class PastRaceBuilder:
    """馬ごとの過去5走成績を計算する（カットオフ厳守・リークなし）"""

    def build(self, df_base: pd.DataFrame) -> pd.DataFrame:
        """
        df_base: race_id, race_date, KettoNum, kakutei_jyuni, time_raw,
                 agari_raw, corner4_raw, syusso_tosu, kyori, odds_tansho 等
        Returns: past_1_rank_rate 〜 past_5_rank_rate 等の特徴量を追加した DataFrame
        """
        logger.info("過去5走成績を計算中...")

        df = df_base.copy()
        df['race_date_int'] = df['race_date'].astype(str).str[:8].astype(int)
        df['jyuni'] = pd.to_numeric(df['kakutei_jyuni'], errors='coerce').fillna(99).astype(int)
        df['tosu']  = pd.to_numeric(df['syusso_tosu'], errors='coerce').fillna(1)
        df['rank_rate'] = (df['tosu'] + 1 - df['jyuni']) / df['tosu'].clip(lower=1)

        # タイム変換（秒）
        df['time_sec_raw'] = df['time_raw'].apply(self._parse_time)
        df['agari_sec']    = pd.to_numeric(df['agari_raw'], errors='coerce') / 10.0

        # kyori を数値化（DB から文字列で来る場合に対応）
        df['kyori_num'] = pd.to_numeric(df['kyori'], errors='coerce')
        # スピード指数計算（簡易版）
        df['speed_fig'] = df.apply(
            lambda r: self._calc_speed_fig(r['time_sec_raw'], r['kyori_num']), axis=1)

        # 過去走の参照用に horse × date でソート
        df = df.sort_values(['KettoNum', 'race_date_int']).reset_index(drop=True)

        rows = []
        for horse_id, grp in df.groupby('KettoNum'):
            grp = grp.reset_index(drop=True)
            for i, row in grp.iterrows():
                past = grp.iloc[:i]  # このレースより前のレースのみ
                feat = self._extract_past_features(past, horse_id, row)
                rows.append(feat)

        df_past = pd.DataFrame(rows)

        # race_id + KettoNum でマージ
        df_merged = df_base.merge(df_past, on=['race_id', 'KettoNum'], how='left')
        logger.info(f"  過去成績特徴量: {len(df_past.columns)}列")
        return df_merged

    @staticmethod
    def _parse_time(val):
        """タイム文字列を秒に変換（例: '0136.5' → 96.5 秒）"""
        if val is None or str(val).strip() in ('', '0000.0', '00000'):
            return None
        try:
            s = str(val).strip().replace(':', '')
            if len(s) == 6:   # MMSSTT（Tは0.1秒）
                m, s_part, t = int(s[:2]), int(s[2:4]), int(s[4:6])
                return m * 60 + s_part + t * 0.1
            elif '.' in s:
                return float(s)
            return float(s)
        except Exception:
            return None

    @staticmethod
    def _calc_speed_fig(time_sec, kyori):
        """簡易スピード指数: 基準タイム差ベース（Noneを返さずに数値化）"""
        if time_sec is None or not kyori:
            return np.nan
        try:
            kyori = float(kyori)
        except (TypeError, ValueError):
            return np.nan
        if kyori <= 0:
            return np.nan
        # 基準タイム = kyori(m) × 0.07秒/m（芝1200m→84秒が目安）
        base_time = kyori * 0.07
        # スピード指数 = 100 - (actual - base) * 3
        return 100 - (time_sec - base_time) * 3.0

    def _extract_past_features(self, past: pd.DataFrame, horse_id, current_row) -> dict:
        feat = {
            'race_id':   current_row['race_id'],
            'KettoNum':  horse_id,
            'past_runs': len(past),
        }

        # 有効な過去走（欠損除外）
        valid = past[past['jyuni'] < 99].copy()
        feat['n_valid_races'] = len(valid)

        if len(valid) == 0:
            return self._fill_defaults(feat)

        # ── 直近5走ループ ────────────────────────────────────────────────
        for k in range(1, 6):
            if len(valid) >= k:
                r = valid.iloc[-k]
                feat[f'past_{k}_rank_rate']   = float(r.get('rank_rate') or np.nan) if r.get('rank_rate') is not None else np.nan
                feat[f'past_{k}_speed_fig']   = float(r.get('speed_fig') or np.nan) if r.get('speed_fig') is not None else np.nan
                feat[f'past_{k}_agari']       = float(r.get('agari_sec') or np.nan) if r.get('agari_sec') is not None else np.nan
                feat[f'past_{k}_corner1']     = self._safe_corner(r.get('corner1_raw'))
                feat[f'past_{k}_corner4']     = self._safe_corner(r.get('corner4_raw'))
                feat[f'past_{k}_tosu']        = pd.to_numeric(r.get('tosu', np.nan), errors='coerce')
                feat[f'past_{k}_kyori']       = pd.to_numeric(r.get('kyori', np.nan), errors='coerce')
                feat[f'past_{k}_bataiju']     = pd.to_numeric(r.get('bataiju', np.nan), errors='coerce')
                # 過去オッズはOK（確定済み過去データ）
                feat[f'past_{k}_odds']        = pd.to_numeric(r.get('odds_tansho', np.nan), errors='coerce')
            else:
                for col in ['rank_rate','speed_fig','agari','corner1','corner4',
                            'tosu','kyori','bataiju']:
                    feat[f'past_{k}_{col}'] = np.nan
                feat[f'past_{k}_odds'] = np.nan

        # ── 集計特徴量 ───────────────────────────────────────────────────
        recent3 = valid.tail(3)
        recent5 = valid.tail(5)

        feat['avg_rank_rate_3']  = recent3['rank_rate'].mean()
        feat['best_rank_rate_5'] = recent5['rank_rate'].max()
        feat['win_rate_5']       = (valid.tail(5)['jyuni'] == 1).mean()
        feat['top3_rate_5']      = (valid.tail(5)['jyuni'] <= 3).mean()
        feat['speed_fig_last']   = valid['speed_fig'].iloc[-1]
        feat['speed_fig_avg3']   = recent3['speed_fig'].mean()
        feat['speed_fig_best3']  = recent5['speed_fig'].nlargest(3).mean()

        sf_vals = recent5['speed_fig'].dropna().tolist()
        feat['speed_fig_trend']  = linear_slope(sf_vals)
        feat['last3_avg_agari']  = recent3['agari_sec'].mean()
        feat['avg_corner4']      = valid['corner4_raw'].apply(
                                        lambda v: self._safe_corner(v)).mean()
        feat['corner4_std']      = valid['corner4_raw'].apply(
                                        lambda v: self._safe_corner(v)).std()

        # 休養期間
        if len(valid) >= 1 and 'race_date_int' in valid.columns:
            last_date = str(valid['race_date_int'].iloc[-1])
            cur_date  = str(current_row['race_date_int'])
            try:
                d1 = datetime.strptime(last_date[:8], '%Y%m%d')
                d2 = datetime.strptime(cur_date[:8], '%Y%m%d')
                feat['interval_days'] = (d2 - d1).days
            except Exception:
                feat['interval_days'] = 90
        else:
            feat['interval_days'] = 365

        feat['is_rest_over_60']  = int(feat.get('interval_days', 0) > 60)
        feat['is_rest_over_90']  = int(feat.get('interval_days', 0) > 90)
        feat['is_rest_over_180'] = int(feat.get('interval_days', 0) > 180)

        # 着差（ranktrend）
        if len(valid) >= 2:
            jyunis = valid['jyuni'].tail(3).tolist()
            feat['rank_trend'] = linear_slope(jyunis)
        else:
            feat['rank_trend'] = 0.0

        # 過去オッズ平均
        past_odds = [feat.get(f'past_{k}_odds') for k in range(1,4)
                     if feat.get(f'past_{k}_odds') and feat[f'past_{k}_odds'] > 0]
        feat['past_1_odds']    = feat.get('past_1_odds', np.nan)
        feat['past_odds_avg3'] = np.mean(past_odds) if past_odds else np.nan

        # 人気比較: 過去に人気以上の着順を出した割合
        hbm_scores = []
        for k in range(1, 6):
            odds_k = feat.get(f'past_{k}_odds', np.nan)
            rr_k   = feat.get(f'past_{k}_rank_rate', np.nan)
            if not pd.isna(odds_k) and not pd.isna(rr_k) and odds_k > 0:
                implied_prob = 1.0 / odds_k
                hbm_scores.append(1 if rr_k > implied_prob else 0)
        feat['hist_beat_market_rate'] = np.mean(hbm_scores) if hbm_scores else 0.5

        # 穴馬での勝利率
        if len(valid) > 0:
            # 過去オッズ5倍超での勝利
            odds_col = pd.to_numeric(valid['odds_tansho'], errors='coerce').fillna(0) if 'odds_tansho' in valid.columns else pd.Series(0, index=valid.index)
            high_odds_wins = valid[odds_col > 5.0]
            feat['underdog_win_rate'] = (
                (high_odds_wins['jyuni'] == 1).sum() / max(len(valid), 1)
            )
        else:
            feat['underdog_win_rate'] = 0.0

        # 着差
        feat['near_miss_flag'] = 0.0
        feat['avg_close_miss'] = 0.0

        # スピード指数モメンタム（ver8新規）
        sf5 = [feat.get(f'past_{k}_speed_fig') for k in range(1,6)]
        sf5_valid = [v for v in sf5 if v is not None and not pd.isna(v)]
        if len(sf5_valid) >= 3:
            feat['sf_slope_5']       = linear_slope(sf5_valid)
            midpoint = len(sf5_valid) // 2
            slope_first = linear_slope(sf5_valid[:midpoint+1])
            slope_second = linear_slope(sf5_valid[midpoint:])
            feat['sf_acceleration'] = slope_second - slope_first
            feat['peak_vs_recent']  = max(sf5_valid) - sf5_valid[-1]
            feat['sf_consistency']  = 1.0 / (np.std(sf5_valid) + 0.1)
        else:
            feat['sf_slope_5']      = 0.0
            feat['sf_acceleration'] = 0.0
            feat['peak_vs_recent']  = 0.0
            feat['sf_consistency']  = 1.0

        # SFピークからの日数
        if len(sf5_valid) > 0 and 'race_date_int' in valid.columns:
            peak_idx = np.argmax(sf5_valid)
            try:
                peak_row = valid.iloc[-(len(sf5_valid) - peak_idx)]
                pd_str   = str(peak_row['race_date_int'])[:8]
                cd_str   = str(current_row['race_date_int'])[:8]
                d1 = datetime.strptime(pd_str, '%Y%m%d')
                d2 = datetime.strptime(cd_str, '%Y%m%d')
                feat['days_since_peak_sf'] = (d2 - d1).days
            except Exception:
                feat['days_since_peak_sf'] = 365
        else:
            feat['days_since_peak_sf'] = 365

        # 馬体重トレンド（過去）
        bataiju_vals = [feat.get(f'past_{k}_bataiju') for k in range(1,4)]
        bataiju_vals = [v for v in bataiju_vals if v is not None and not pd.isna(v)]
        if bataiju_vals:
            feat['bataiju_avg3']  = np.mean(bataiju_vals)
            feat['bataiju_trend'] = linear_slope(bataiju_vals)
            feat['bataiju_std']   = np.std(bataiju_vals)
        else:
            feat['bataiju_avg3']  = np.nan
            feat['bataiju_trend'] = 0.0
            feat['bataiju_std']   = np.nan

        # フォームサイクル判定
        slope = feat.get('sf_slope_5', 0)
        if slope > FeatureConfig.FORM_SLOPE_UP:
            feat['form_cycle_stage'] = 'up'
        elif slope < FeatureConfig.FORM_SLOPE_DOWN:
            feat['form_cycle_stage'] = 'down'
        else:
            feat['form_cycle_stage'] = 'peak'

        return feat

    @staticmethod
    def _safe_corner(val):
        return parse_corner(val)

    def _fill_defaults(self, feat):
        for k in range(1, 6):
            for col in ['rank_rate','speed_fig','agari','corner1','corner4',
                        'tosu','kyori','bataiju','odds']:
                feat[f'past_{k}_{col}'] = np.nan
        for col in ['avg_rank_rate_3','best_rank_rate_5','win_rate_5','top3_rate_5',
                    'speed_fig_last','speed_fig_avg3','speed_fig_best3','speed_fig_trend',
                    'last3_avg_agari','avg_corner4','corner4_std','interval_days',
                    'rank_trend','past_1_odds','past_odds_avg3',
                    'hist_beat_market_rate','underdog_win_rate','near_miss_flag',
                    'avg_close_miss','sf_slope_5','sf_acceleration','peak_vs_recent',
                    'sf_consistency','days_since_peak_sf','bataiju_avg3','bataiju_trend',
                    'bataiju_std']:
            feat[col] = 0.0 if col != 'past_1_odds' else np.nan
        feat['is_rest_over_60']  = 0
        feat['is_rest_over_90']  = 0
        feat['is_rest_over_180'] = 0
        feat['form_cycle_stage'] = 'unknown'
        return feat


# =============================================================================
# 脚質・展開特徴量
# =============================================================================
class StyleFeatureBuilder:
    """脚質・ペース特徴量を計算する（过去の成績のみ参照）"""

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("脚質・展開特徴量を計算中...")
        df = df.copy()

        # DB 取得列を数値化（SQLite の文字列型対策）
        df['kyori']        = pd.to_numeric(df['kyori'],        errors='coerce')
        df['syusso_tosu']  = pd.to_numeric(df['syusso_tosu'],  errors='coerce')

        # KyakuShitsu（脚質）: 過去のコーナー4番ポジションから推定
        df['KyakuShitsu'] = df.apply(
            lambda r: classify_kyakushitsu(
                r.get('avg_corner4'), r.get('syusso_tosu')), axis=1)

        # 逃げ先行頭数（レース単位で集計）
        def race_front_runners(grp):
            n_front = (grp['KyakuShitsu'] <= 2).sum()
            grp['n_front_runners']   = n_front
            grp['front_runner_ratio'] = n_front / max(len(grp), 1)
            return grp

        df = df.groupby('race_id', group_keys=False).apply(race_front_runners)

        # ペースカテゴリ
        def pace_cat(row):
            r = row.get('front_runner_ratio', 0)
            if r > 0.40: return 'high'
            elif r > 0.20: return 'mid'
            else: return 'slow'

        df['pace_category'] = df.apply(pace_cat, axis=1)

        # ペース競合度スコア
        df['pace_pressure'] = df['n_front_runners'] * (df['kyori'].fillna(1800) / 1800)

        # 自馬脚質とペースの適合度
        def style_fit(row):
            ks = row.get('KyakuShitsu', 3)
            pc = row.get('pace_category', 'mid')
            if ks == 4 and pc == 'high':   return 1.0
            if ks == 1 and pc == 'slow':   return 1.0
            if ks in (1,2) and pc == 'high': return -0.5
            if ks == 4 and pc == 'slow':   return -0.5
            return 0.0

        df['style_fit_score'] = df.apply(style_fit, axis=1)

        # ten_speed（前半速度）
        df['ten_speed'] = df.apply(
            lambda r: calc_ten_speed(None, r.get('last3_avg_agari'), r.get('kyori')),
            axis=1)
        df['ten_speed'] = df['ten_speed'].fillna(df['ten_speed'].median())

        df['pace_pred']    = df['pace_category'].map({'high':1,'mid':0,'slow':-1}).fillna(0)
        df['pace_tekisei'] = df['style_fit_score'].fillna(0)

        df['pos_gain_c1_c4']  = np.nan
        df['past_avg_pos_gain'] = np.nan

        logger.info("  脚質特徴量完了")
        return df


# =============================================================================
# 週別成績計算クラス（リークなし累積集計）
# =============================================================================
class WeeklyStatsBuilder:
    """
    各レースの「開催週月曜日」をカットオフとして、
    それ以前の全レース結果から累積成績を計算する。
    （同週の他レース結果はリーク防止のため含まない）
    """

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("週別累積成績を計算中...")

        df = df.copy()
        df['race_date_int'] = df['race_date'].astype(str).str[:8].astype(int)
        df['monday']        = df['race_date'].astype(str).str[:8].apply(prev_week_monday)
        df['win_flag']      = (pd.to_numeric(df['kakutei_jyuni'], errors='coerce') == 1).astype(int)
        df['top3_flag']     = (pd.to_numeric(df['kakutei_jyuni'], errors='coerce') <= 3).astype(int)
        df['track_type']    = df['TrackCD'].apply(CodeMaster.get_track_type)

        unique_mondays = sorted(df['monday'].unique())

        # ─── 騎手成績 ────────────────────────────────────────────────────
        jockey_stats = self._compute_entity_stats(
            df, 'KisyuCode', unique_mondays,
            out_cols={'win_rate': 'jockey_win_rate', 'top3_rate': 'jockey_top3_rate'})

        # ─── 調教師成績 ──────────────────────────────────────────────────
        trainer_stats = self._compute_entity_stats(
            df, 'ChokyosiCode', unique_mondays,
            out_cols={'win_rate': 'trainer_win_rate', 'top3_rate': 'trainer_top3_rate'})

        # ─── 種牡馬成績 ──────────────────────────────────────────────────
        sire_stats = self._compute_entity_stats(
            df, 'SireID', unique_mondays,
            out_cols={'win_rate': 'sire_win_rate', 'top3_rate': 'sire_top3_rate'})

        # ─── 騎手×調教師コンボ成績 ───────────────────────────────────────
        combo_stats = self._compute_combo_stats(df, unique_mondays)

        # ─── 騎手の直近14日勝率 ──────────────────────────────────────────
        recent_jockey = self._compute_recent_stats(df, 'KisyuCode', days=14)

        # ─── df に結合 ────────────────────────────────────────────────────
        df = df.merge(jockey_stats,  on=['monday', 'KisyuCode'], how='left')
        df = df.merge(trainer_stats, on=['monday', 'ChokyosiCode'], how='left')
        df = df.merge(sire_stats,    on=['monday', 'SireID'], how='left')
        df = df.merge(combo_stats,   on=['race_id', 'KettoNum'], how='left')
        df = df.merge(recent_jockey, on=['race_id', 'KisyuCode'], how='left')

        df['jockey_win_rate']  = df['jockey_win_rate'].fillna(0.1)
        df['trainer_win_rate'] = df['trainer_win_rate'].fillna(0.05)
        df['sire_win_rate']    = df['sire_win_rate'].fillna(0.05)

        logger.info("  週別成績完了")
        return df

    def _compute_entity_stats(self, df, entity_col, mondays, out_cols):
        """カットオフ厳守の累積成績計算（汎用版）"""
        df_e = df[[entity_col, 'race_date_int', 'win_flag', 'top3_flag', 'monday']].copy()

        rows = []
        for monday in mondays:
            cutoff = int(monday)
            sub = df_e[df_e['race_date_int'] < cutoff]
            if len(sub) == 0:
                continue
            agg = sub.groupby(entity_col).agg(
                runs=('win_flag', 'count'),
                wins=('win_flag', 'sum'),
                top3=('top3_flag', 'sum')
            ).reset_index()
            agg['win_rate']  = agg['wins'] / agg['runs'].clip(lower=1)
            agg['top3_rate'] = agg['top3'] / agg['runs'].clip(lower=1)
            agg['monday']    = monday

            # カラム名を変換
            rename = {k: v for k, v in out_cols.items()}
            agg = agg.rename(columns=rename)
            rows.append(agg[[entity_col, 'monday'] + list(out_cols.values())])

        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
            columns=[entity_col, 'monday'] + list(out_cols.values()))

    def _compute_combo_stats(self, df, mondays):
        """騎手×調教師コンボ成績（カットオフ厳守）"""
        df_c = df[['KisyuCode','ChokyosiCode','race_date_int','win_flag','top3_flag',
                   'race_id','KettoNum','monday']].copy()

        result_rows = []
        for monday in mondays:
            cutoff = int(monday)
            sub = df_c[df_c['race_date_int'] < cutoff]
            if len(sub) == 0:
                continue
            agg = sub.groupby(['KisyuCode','ChokyosiCode']).agg(
                runs=('win_flag','count'),
                wins=('win_flag','sum'),
                top3=('top3_flag','sum')
            ).reset_index()
            agg['trainer_jockey_combo_win']  = agg['wins'] / agg['runs'].clip(lower=1)
            agg['trainer_jockey_combo_top3'] = agg['top3'] / agg['runs'].clip(lower=1)
            agg['trainer_jockey_runs']       = agg['runs']

            # その月曜日に対応するレース
            race_sub = df_c[df_c['monday'] == monday][
                ['race_id','KettoNum','KisyuCode','ChokyosiCode']].drop_duplicates()
            merged = race_sub.merge(
                agg[['KisyuCode','ChokyosiCode',
                     'trainer_jockey_combo_win','trainer_jockey_combo_top3',
                     'trainer_jockey_runs']],
                on=['KisyuCode','ChokyosiCode'], how='left')
            result_rows.append(merged)

        if result_rows:
            return pd.concat(result_rows, ignore_index=True)[
                ['race_id','KettoNum',
                 'trainer_jockey_combo_win','trainer_jockey_combo_top3',
                 'trainer_jockey_runs']].drop_duplicates(['race_id','KettoNum'])
        else:
            return pd.DataFrame(columns=['race_id','KettoNum',
                'trainer_jockey_combo_win','trainer_jockey_combo_top3',
                'trainer_jockey_runs'])

    def _compute_recent_stats(self, df, entity_col, days=14):
        """直近N日の騎手成績（カットオフ厳守・高速版: エンティティ別にグループ化して O(G×g²)）"""
        df_r = df[[entity_col, 'race_id', 'race_date_int', 'win_flag']].copy()

        results = []
        for entity_val, grp in df_r.groupby(entity_col):
            grp   = grp.sort_values('race_date_int').reset_index(drop=True)
            dates = grp['race_date_int'].values
            wins  = grp['win_flag'].values.astype(float)

            recent_wrs = []
            for i in range(len(grp)):
                cutoff = int(dates[i])
                try:
                    d      = datetime.strptime(str(cutoff)[:8], '%Y%m%d')
                    start  = int((d - timedelta(days=days)).strftime('%Y%m%d'))
                except Exception:
                    recent_wrs.append(np.nan)
                    continue
                mask = (dates >= start) & (dates < cutoff)
                recent_wrs.append(float(wins[mask].mean()) if mask.any() else np.nan)

            grp['jockey_recent_win_rate'] = recent_wrs
            results.append(grp[['race_id', entity_col, 'jockey_recent_win_rate']])

        if results:
            out = pd.concat(results, ignore_index=True)
            # 同一 (race_id, entity_col) が複数行ある場合（null コード等）は先頭を残す
            out = out.drop_duplicates(subset=['race_id', entity_col], keep='first')
            return out
        return pd.DataFrame(columns=['race_id', entity_col,
                                     'jockey_recent_win_rate'])


# =============================================================================
# コース固有適性特徴量
# =============================================================================
class CourseAffinityBuilder:
    """同コース×距離での過去成績を計算（カットオフ厳守・cumsum 高速版）"""

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("コース固有適性を計算中...")
        df = df.copy()

        # kyori を数値化（未変換の場合に備えて）
        df['kyori'] = pd.to_numeric(df['kyori'], errors='coerce')

        df['dist_bucket'] = pd.cut(
            df['kyori'].fillna(1800), bins=FeatureConfig.DIST_BUCKET_BINS,
            labels=FeatureConfig.DIST_BUCKET_LABELS, right=False).astype(str)

        if 'race_date_int' not in df.columns:
            df['race_date_int'] = df['race_date'].astype(str).str[:8].astype(int)

        # win/top3 フラグ（kakutei_jyuni から直接生成）
        jyuni_num = pd.to_numeric(df['kakutei_jyuni'], errors='coerce')
        df['_wfn'] = (jyuni_num == 1).fillna(False).astype(int)
        df['_t3n'] = (jyuni_num <= 3).fillna(False).astype(int)

        # ── 高速版: グループ内 cumsum でカットオフ前の成績を計算 ────────────
        # 同じ馬・競馬場・距離区分でソートして cumcount / cumsum を利用
        group_cols = ['KettoNum', 'JyoCD', 'dist_bucket']
        sort_cols  = group_cols + ['race_date_int']

        df_s = df[['race_id','KettoNum','JyoCD','dist_bucket',
                   'race_date_int','_wfn','_t3n']].sort_values(sort_cols).copy()

        # 同グループ内の位置（前走数）= cumcount（0始まり）
        df_s['course_runs'] = df_s.groupby(group_cols, sort=False).cumcount()
        # 前走までの累計勝利・複勝 = cumsum - 当該レース分
        df_s['_cw']  = (df_s.groupby(group_cols, sort=False)['_wfn']
                        .cumsum() - df_s['_wfn'])
        df_s['_ct3'] = (df_s.groupby(group_cols, sort=False)['_t3n']
                        .cumsum() - df_s['_t3n'])

        df_s['course_win_rate'] = np.where(
            df_s['course_runs'] > 0, df_s['_cw'] / df_s['course_runs'], 0.0)
        df_s['course_top3_rate'] = np.where(
            df_s['course_runs'] > 0, df_s['_ct3'] / df_s['course_runs'], 0.0)

        # race_id + KettoNum でマージ
        df = df.merge(
            df_s[['race_id','KettoNum','course_win_rate','course_top3_rate','course_runs']],
            on=['race_id','KettoNum'], how='left')

        df['course_win_rate']  = df['course_win_rate'].fillna(0.0)
        df['course_top3_rate'] = df['course_top3_rate'].fillna(0.0)
        df['course_runs']      = df['course_runs'].fillna(0)

        # 一時列を削除
        df = df.drop(columns=['_wfn', '_t3n'], errors='ignore')

        logger.info("  コース固有適性完了")
        return df


# =============================================================================
# ターゲット変数計算
# =============================================================================
class TargetBuilder:
    """
    学習用ターゲット変数を計算する。
    確定済み歴史データ（払戻金）を使用するため学習時のみOK。
    """

    def build(self, df: pd.DataFrame, df_payout: pd.DataFrame) -> pd.DataFrame:
        logger.info("ターゲット変数を計算中...")
        df = df.copy()

        # 払戻データを結合（race_id × umaban レベルで horse ごとに結合）
        if len(df_payout) > 0 and 'umaban' in df_payout.columns:
            # umaban を int 型に統一してから結合
            df['_umaban_int'] = pd.to_numeric(df['umaban'], errors='coerce').fillna(0).astype(int)
            df_pay2 = df_payout.copy()
            df_pay2['umaban'] = pd.to_numeric(df_pay2['umaban'], errors='coerce').fillna(0).astype(int)
            # N_HARAI に重複行がある場合を排除（(race_id, umaban) は一意になるべき）
            df_pay2 = df_pay2.drop_duplicates(subset=['race_id', 'umaban'], keep='first')
            df = df.merge(df_pay2, left_on=['race_id', '_umaban_int'],
                          right_on=['race_id', 'umaban'], how='left',
                          suffixes=('', '_pay'))
            # マージ後の不要列を除去
            drop_cols = ['_umaban_int']
            if 'umaban_pay' in df.columns:
                drop_cols.append('umaban_pay')
            df = df.drop(columns=drop_cols, errors='ignore')
            # レースレベルの最高単勝払戻（荒れ度ターゲット計算用）
            race_max_pay = (df_pay2.groupby('race_id')['tansho_payout']
                            .max().reset_index()
                            .rename(columns={'tansho_payout': '_race_tansho_pay'}))
            df = df.merge(race_max_pay, on='race_id', how='left')
        else:
            df['tansho_payout']    = np.nan
            df['fukusho_payout']   = np.nan
            df['_race_tansho_pay'] = np.nan

        df['tansho_payout']  = pd.to_numeric(df['tansho_payout'], errors='coerce')
        df['fukusho_payout'] = pd.to_numeric(df['fukusho_payout'], errors='coerce')
        df['jyuni_num']      = pd.to_numeric(df['kakutei_jyuni'], errors='coerce').fillna(99)

        # ── 単勝EV ──────────────────────────────────────────────────────────
        # target_ev_win = tansho_payout/100 - 1 (的中) / -1 (外れ)
        df['win_flag']  = (df['jyuni_num'] == 1).astype(int)
        df['place_flag'] = (df['jyuni_num'] <= 3).astype(int)

        tansho_payout_norm = df['tansho_payout'].fillna(0) / 100.0
        df['target_ev_win'] = np.where(
            df['win_flag'] == 1, tansho_payout_norm - 1.0, -1.0)
        df['target_ev_win'] = df['target_ev_win'].clip(-1.0, 10.0)

        # ── 複勝EV ──────────────────────────────────────────────────────────
        fukusho_norm = df['fukusho_payout'].fillna(0) / 100.0
        df['target_ev_place'] = np.where(
            df['place_flag'] == 1, fukusho_norm - 1.0, -1.0)
        df['target_ev_place'] = df['target_ev_place'].clip(-1.0, 5.0)

        # ── 勝率ソフトラベル (Win soft) ──────────────────────────────────────
        df['target_win_soft'] = df['win_flag'].astype(float)

        # ── 複勝率ソフトラベル (HitRate) ─────────────────────────────────────
        df['target_hitrate_soft'] = df['place_flag'].astype(float)

        # ── LTR relevance: max(0, EV_win) ────────────────────────────────────
        df['target_ltr_ev'] = df['target_ev_win'].clip(lower=0.0)

        # ── 荒れ度ターゲット ──────────────────────────────────────────────────
        # レース全体の単勝払戻 > TURB_PAY_THR → 荒れレース（全馬共通フラグ）
        df['target_turbulence'] = (
            df['_race_tansho_pay'].fillna(0) > TrainConfig.TURB_PAY_THR).astype(int)
        df = df.drop(columns=['_race_tansho_pay'], errors='ignore')

        n_win  = df['win_flag'].sum()
        n_turb = df['target_turbulence'].sum()
        logger.info(f"  EV_win mean={df['target_ev_win'].mean():.3f}")
        logger.info(f"  win_flag={n_win:,} ({df['win_flag'].mean():.1%})")
        logger.info(f"  turbulence={n_turb:,} ({df['target_turbulence'].mean():.1%})")

        return df


# =============================================================================
# 特徴量エンジニアリング（仕上げ）
# =============================================================================
class FeatureEngineer:
    """数値特徴量の変換・カテゴリ型変換・派生特徴量を追加"""

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("特徴量エンジニアリング（仕上げ）中...")
        df = df.copy()

        # ── カテゴリ変数を str 型に変換（LightGBM native categorical用）─────
        for col in FeatureConfig.CATEGORICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna('unknown').astype(str)

        # ── 派生カテゴリカル ─────────────────────────────────────────────────
        df['kyori_cat'] = pd.cut(
            df['kyori'].fillna(1800), bins=FeatureConfig.KYORI_BINS,
            labels=FeatureConfig.KYORI_LABELS, right=False).astype(str)

        if 'dist_bucket' not in df.columns:
            df['dist_bucket'] = pd.cut(
                df['kyori'].fillna(1800), bins=FeatureConfig.DIST_BUCKET_BINS,
                labels=FeatureConfig.DIST_BUCKET_LABELS, right=False).astype(str)

        df['track_type'] = df['TrackCD'].apply(CodeMaster.get_track_type).astype(str)

        # 休養期間カテゴリ
        def rest_cat(days):
            if days is None or pd.isna(days): return 'unknown'
            days = float(days)
            if days <= 14:   return 'short'
            elif days <= 42: return 'mid'
            elif days <= 90: return 'long'
            else:            return 'very_long'

        df['rest_category'] = df['interval_days'].apply(rest_cat).astype(str)

        # キャリア段階
        def career_stage(runs):
            if runs is None or pd.isna(runs): return 'unknown'
            runs = int(runs)
            if runs <= 3:   return 'early'
            elif runs <= 10: return 'mid'
            else:           return 'veteran'

        df['career_stage'] = df['n_valid_races'].apply(career_stage).astype(str)

        # グレード数値
        df['grade_num'] = df['GradeCD'].map(
            {'A':8,'B':7,'C':6,'D':5,'E':4,'F':3,'G':2,'H':1,'':3}).fillna(3)

        # 出走番号（枠番近似）
        df['wakuban'] = pd.to_numeric(df.get('Wakuban', df.get('wakuban', 0)),
                                      errors='coerce').fillna(0)

        # 距離差・クラス差（前走比）
        df['dist_diff'] = (
            df['kyori'].fillna(1800) - df['past_1_kyori'].fillna(1800))
        df['class_diff'] = 0  # シンプル化

        # 騎手変更フラグ（過去と現在の騎手コード比較 → 簡易版）
        df['jockey_change'] = 0

        # 馬具変化
        if 'blinker_cd' in df.columns:
            df['blinker_on']    = (df['blinker_cd'].astype(str) == '1').astype(int)
            df['blinker_first'] = 0
            df['blinker_off']   = 0
        else:
            df['blinker_on']    = 0
            df['blinker_first'] = 0
            df['blinker_off']   = 0

        # 芝⇔ダート変更
        df['turf_dirt_change'] = 0

        # 斤量差（前走比）
        # futan は DB から文字列で来るため数値化。past_1_futan 特徴量がないため差分は 0 固定
        df['futan'] = pd.to_numeric(df['futan'], errors='coerce')
        df['futan_diff'] = 0.0

        # form_cycle_stage が未設定の場合
        if 'form_cycle_stage' not in df.columns:
            df['form_cycle_stage'] = 'unknown'
        df['form_cycle_stage'] = df['form_cycle_stage'].fillna('unknown').astype(str)

        # pace_category が未設定の場合
        if 'pace_category' not in df.columns:
            df['pace_category'] = 'mid'
        df['pace_category'] = df['pace_category'].fillna('mid').astype(str)

        # 欠損値補完
        for col in FeatureConfig.NUMERIC_FEATURES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        logger.info("  特徴量エンジニアリング完了")
        return df


# =============================================================================
# FORBIDDEN_COLS 安全パージ（最重要ステップ）
# =============================================================================
def purge_forbidden_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    ★★★ 絶対に実行する最終安全装置 ★★★
    当日確定情報・ターゲット変数を特徴量から強制削除する。

    このステップにより:
    - odds_tansho, tansho_ninki, bataiju, zogen_sa → 削除（当日情報）
    - kakutei_jyuni, time, agari → 削除（当日情報）
    - win_flag, target_ev_win など → 削除（目的変数はdf_targetに分離）
    """
    before = set(df.columns)
    df_clean = df.drop(columns=FORBIDDEN_FEATURE_COLS, errors='ignore')
    removed = before - set(df_clean.columns)

    if removed:
        logger.info(f"\n[FORBIDDEN_COLS パージ完了] {len(removed)}列を削除:")
        for c in sorted(removed):
            logger.info(f"  X {c}")
    else:
        logger.info("[FORBIDDEN_COLS パージ] 削除対象列なし（すでにクリーン）")

    return df_clean


# =============================================================================
# メイン処理
# =============================================================================
def main():
    setup_logging()
    Config.init()

    logger.info("=" * 70)
    logger.info("ver8 Step 3: 学習データ作成開始")
    logger.info("=" * 70)

    db = DBConnector()
    loader = RaceDataLoader(db)

    try:
        # ─── 1. ベースデータ読み込み ──────────────────────────────────────
        logger.info("\n[1/8] ベースデータ読み込み...")
        df_base   = loader.load_base(start_year=2016, end_year=2026)
        df_payout = loader.load_payouts(start_year=2016, end_year=2026)
        df_kaisai = loader.load_kaisai(start_year=2016, end_year=2026)

        # 開催データ結合
        if len(df_kaisai) > 0:
            df_base['kaisai_key'] = (df_base['race_date'].astype(str).str[:8] +
                                      df_base['JyoCD'].astype(str))
            df_base = df_base.merge(
                df_kaisai.rename(columns={
                    'TenkoCD': 'TenkoCD_k', 'SibaBabaCD': 'SibaBabaCD_k',
                    'DirtBabaCD': 'DirtBabaCD_k',
                }), on='kaisai_key', how='left')
            # kaisai の天候・馬場が優先（N_RACE のものより精度が高い場合がある）
            for col in ['TenkoCD','SibaBabaCD','DirtBabaCD']:
                k_col = f'{col}_k'
                if k_col in df_base.columns:
                    df_base[col] = df_base.get(k_col, df_base.get(col))
            df_base['track_bias_inner'] = df_base.get('track_bias_inner', 0)
            df_base['track_bias_pace']  = df_base.get('track_bias_pace', 0)

        # ─── 2. フィルタリング ─────────────────────────────────────────────
        logger.info("\n[2/8] フィルタリング...")
        n_before = len(df_base)

        if TrainConfig.EXCLUDE_OBSTACLE:
            df_base = df_base[~df_base['TrackCD'].astype(str).str.startswith('5')]
        if TrainConfig.EXCLUDE_SHINBA:
            df_base = df_base[~(df_base['GradeCD'].astype(str) == '')]

        df_base = df_base[
            pd.to_numeric(df_base['syusso_tosu'], errors='coerce').fillna(0) >= TrainConfig.MIN_HORSES
        ]

        logger.info(f"  フィルタリング: {n_before:,} → {len(df_base):,} 行")

        # ─── 3. 過去5走成績 ────────────────────────────────────────────────
        logger.info("\n[3/8] 過去5走成績計算...")
        past_builder = PastRaceBuilder()
        df_base = past_builder.build(df_base)
        logger.info(f"  [行数チェック] 3/8後: {len(df_base):,} 行")

        # ─── 4. 脚質・展開特徴量 ───────────────────────────────────────────
        logger.info("\n[4/8] 脚質・展開特徴量...")
        style_builder = StyleFeatureBuilder()
        df_base = style_builder.build(df_base)
        logger.info(f"  [行数チェック] 4/8後: {len(df_base):,} 行")

        # ─── 5. 週別累積成績（リークなし）─────────────────────────────────
        logger.info("\n[5/8] 週別累積成績（カットオフ厳守）...")
        weekly_builder = WeeklyStatsBuilder()
        df_base = weekly_builder.build(df_base)
        logger.info(f"  [行数チェック] 5/8後: {len(df_base):,} 行")

        # ─── 6. コース固有適性 ────────────────────────────────────────────
        logger.info("\n[6/8] コース固有適性...")
        course_builder = CourseAffinityBuilder()
        df_base = course_builder.build(df_base)
        logger.info(f"  [行数チェック] 6/8後: {len(df_base):,} 行")

        # ─── 7. ターゲット変数計算 ────────────────────────────────────────
        logger.info("\n[7/8] ターゲット変数計算...")
        target_builder = TargetBuilder()
        df_base = target_builder.build(df_base, df_payout)
        logger.info(f"  [行数チェック] 7/8後: {len(df_base):,} 行")

        # ─── 8. 特徴量エンジニアリング（仕上げ） ────────────────────────
        logger.info("\n[8/8] 特徴量エンジニアリング（仕上げ）...")
        fe = FeatureEngineer()
        df_base = fe.build(df_base)

        # ─── ★★★ FORBIDDEN_COLS 安全パージ（最終ステップ）★★★ ────────────
        logger.info("\n[PURGE] FORBIDDEN_COLS 安全パージ実行...")

        # ターゲット変数は別途保持
        target_cols  = [c for c in [
            TrainConfig.TARGET_EV_WIN, TrainConfig.TARGET_EV_PLACE,
            TrainConfig.TARGET_WIN_SOFT, TrainConfig.TARGET_HITRATE,
            TrainConfig.TARGET_LTR, TrainConfig.TARGET_TURB,
            'win_flag', 'place_flag'
        ] if c in df_base.columns]

        id_cols = ['race_id', 'race_date', 'year', 'KettoNum', 'umaban', 'JyoCD']

        # ターゲット列を一時退避
        df_targets = df_base[
            [c for c in id_cols + target_cols if c in df_base.columns]
        ].copy()

        # FORBIDDEN_COLS 強制削除
        df_features = purge_forbidden_cols(df_base)

        # ─── 特徴量リスト確認 ────────────────────────────────────────────
        all_feats = FeatureConfig.get_all_features()
        available = [c for c in all_feats if c in df_features.columns]
        missing   = [c for c in all_feats if c not in df_features.columns]

        logger.info(f"\n[特徴量サマリ]")
        logger.info(f"  利用可能: {len(available)}/{len(all_feats)} 列")
        if missing:
            logger.warning(f"  未生成（デフォルト0で補完）: {len(missing)} 列")
            for c in missing[:20]:
                logger.warning(f"    - {c}")
            # 欠損特徴量はNaN埋め
            for c in missing:
                df_features[c] = np.nan

        # ─── 年別に保存 ──────────────────────────────────────────────────
        df_features['race_date'] = df_features['race_date'].astype(str)
        df_features['year']      = df_features['race_date'].str[:4].astype(int)

        logger.info("\n[年別保存]")
        for year in sorted(df_features['year'].unique()):
            df_yr = df_features[df_features['year'] == year].copy()
            df_tg = df_targets[df_targets['year'].astype(int) == year].copy() \
                    if 'year' in df_targets.columns else df_targets.copy()

            # 特徴量 + ターゲットを結合して保存
            save_cols = id_cols + available
            df_save = df_yr[[c for c in save_cols if c in df_yr.columns]].copy()
            df_save = df_save.merge(
                df_tg[[c for c in id_cols + target_cols if c in df_tg.columns]],
                on=[c for c in id_cols if c in df_tg.columns and c in df_save.columns],
                how='left')

            path = f"{PathConfig.DATA_DIR}/features_{year}.parquet"
            df_save.to_parquet(path, index=False)
            logger.info(f"  {year}年: {len(df_save):,} 行 → {path}")

        # カテゴリ特徴量リストを保存
        cat_features = [c for c in FeatureConfig.get_all_categoricals()
                        if c in df_features.columns]
        with open(f'{PathConfig.MASTER_DIR}/cat_features_v8.pkl', 'wb') as f:
            pickle.dump(cat_features, f)
        logger.info(f"\nカテゴリ特徴量リスト保存: {len(cat_features)}列")

        # 特徴量リスト保存
        with open(f'{PathConfig.MASTER_DIR}/feature_cols_v8.pkl', 'wb') as f:
            pickle.dump(available, f)
        logger.info(f"特徴量リスト保存: {len(available)}列")

        logger.info("\n" + "=" * 70)
        logger.info("Step 3 完了")
        logger.info("=" * 70)

    finally:
        db.close()


if __name__ == '__main__':
    main()
