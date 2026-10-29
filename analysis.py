"""기상 조건이 섬 관광지 입도 관광객에 미치는 영향 분석.

실행:  python analysis.py
입력:  data/visitors_daily.csv, data/weather_daily.csv, data/flights_monthly.csv
출력:  images/*.png, results/metrics.json, dashboard/data.js
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).parent
DATA, IMG, RES, DASH = ROOT / "data", ROOT / "images", ROOT / "results", ROOT / "dashboard"
for d in (IMG, RES, DASH):
    d.mkdir(exist_ok=True)

# ---------------------------------------------------------------- 스타일
for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
          "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
    if Path(p).exists():
        font_manager.fontManager.addfont(p)
_avail = {f.name for f in font_manager.fontManager.ttflist}
_FONTS = [n for n in ("Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic", "Malgun Gothic", "AppleGothic")
          if n in _avail] + ["DejaVu Sans"]
plt.rcParams.update({
    "font.family": _FONTS,
    "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#B9B7AD", "axes.grid": True, "grid.color": "#EDEDED", "figure.dpi": 130,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})
BLUE, ORANGE, GRAY, GREEN, PINK, PURPLE = "#2B79D8", "#EC6A36", "#8A8880", "#1AA06D", "#E377A3", "#4B35A8"

# ---------------------------------------------------------------- 설정값
BASE_WIN = 31          # 기대치 기준선: 요일 보정 이동중앙값 창(일)
GUST_WEAK = 18.0       # 약한 기상 기준: 순간풍속(m/s)
GUST_WARN = 20.0       # 강풍주의보 기준: 순간풍속 초과(m/s)
SNOW_WARN = 5.0        # 대설주의보 기준 근사: 일 최심신적설(cm)
RAIN_WARN = 80.0       # 호우주의보 기준 근사: 일강수량(mm)
CLUSTER_GAP = 2        # 충격일이 이 간격 이내면 한 사건으로 묶음
SEED = 42

# 공휴일·연휴 블록(공휴일, 대체·임시공휴일, 맞닿은 주말 포함, 직접 정리한 표)
HOLIDAY_BLOCKS = [
    # 2023
    ("2023-01-01", "2023-01-01"), ("2023-01-21", "2023-01-24"), ("2023-03-01", "2023-03-01"),
    ("2023-05-05", "2023-05-07"), ("2023-05-27", "2023-05-29"), ("2023-06-06", "2023-06-06"),
    ("2023-08-12", "2023-08-15"), ("2023-09-28", "2023-10-03"), ("2023-10-07", "2023-10-09"),
    ("2023-12-23", "2023-12-25"),
    # 2024
    ("2024-01-01", "2024-01-01"), ("2024-02-09", "2024-02-12"), ("2024-03-01", "2024-03-03"),
    ("2024-04-10", "2024-04-10"), ("2024-05-04", "2024-05-06"), ("2024-05-15", "2024-05-15"),
    ("2024-06-06", "2024-06-09"), ("2024-08-15", "2024-08-18"), ("2024-09-14", "2024-09-18"),
    ("2024-10-03", "2024-10-06"), ("2024-10-09", "2024-10-09"), ("2024-12-25", "2024-12-25"),
    # 2025
    ("2025-01-01", "2025-01-01"), ("2025-01-25", "2025-01-30"), ("2025-03-01", "2025-03-03"),
    ("2025-05-03", "2025-05-06"), ("2025-06-03", "2025-06-03"), ("2025-06-06", "2025-06-08"),
    ("2025-08-15", "2025-08-17"), ("2025-10-03", "2025-10-09"), ("2025-12-25", "2025-12-25"),
    # 2026
    ("2026-01-01", "2026-01-01"), ("2026-02-14", "2026-02-18"), ("2026-03-01", "2026-03-02"),
    ("2026-05-02", "2026-05-05"), ("2026-05-23", "2026-05-25"), ("2026-06-03", "2026-06-03"),
    ("2026-06-06", "2026-06-06"), ("2026-08-15", "2026-08-17"), ("2026-09-24", "2026-09-28"),
]

CAUSES = {"W_WARN": "기상: 특보 기준 이상", "W_WEAK": "기상: 순간풍속 18m/s+·적설",
          "HOL": "연휴·명절 여파", "SEM": "3월 개학기", "UNK": "원인 미확인"}
CAUSE_COLOR = {"W_WARN": BLUE, "W_WEAK": GREEN, "HOL": ORANGE, "SEM": PINK, "UNK": PURPLE}


# ================================================================= 1. 데이터 로드·점검
def load():
    v = pd.read_csv(DATA / "visitors_daily.csv", parse_dates=["date"])
    w = pd.read_csv(DATA / "weather_daily.csv", parse_dates=["date"])
    f = pd.read_csv(DATA / "flights_monthly.csv")
    df = v.merge(w, on="date", how="left").sort_values("date").set_index("date")
    full = pd.date_range(df.index.min(), df.index.max(), freq="D")
    info = {
        "rows": int(len(df)), "start": str(df.index.min().date()), "end": str(df.index.max().date()),
        "missing_dates": int(len(full.difference(df.index))),
        "null_cells": {c: int(df[c].isna().sum()) for c in df.columns},
        "snow_days": int((df["snow_cm"] > 0).sum()), "rain_days": int((df["precip_mm"] > 0).sum()),
        "flight_months": int(len(f)),
    }
    return df, f, info


def holiday_sets(index):
    hol = set()
    after = set()
    for s, e in HOLIDAY_BLOCKS:
        days = pd.date_range(s, e)
        hol.update(days)
        after.update(pd.date_range(days[0], days[-1] + pd.Timedelta(days=2)))
    return hol, after


# ================================================================= 2. 기대치·편차율
def build_expected(df, win=BASE_WIN):
    """요일 보정 이동중앙값 기대치. 연휴·강한 기상일은 기준선 계산에서 뺀다."""
    hol, _ = holiday_sets(df.index)
    v = df["domestic"].astype(float)
    gust2 = np.maximum(df["gust_ms"], df["gust_ms"].shift(1).fillna(df["gust_ms"]))
    excl = pd.Series([d in hol for d in df.index], index=df.index) | (gust2 >= GUST_WEAK) | \
        (df["snow_cm"] > 0) | (df["precip_mm"] >= RAIN_WARN)
    vm = v.where(~excl)
    base0 = vm.rolling(win, center=True, min_periods=win // 2).median()
    ratio = (vm / base0).dropna()
    wd_factor = ratio.groupby(ratio.index.dayofweek).median()
    wd_factor = wd_factor / wd_factor.mean()
    adj = (vm / v.index.dayofweek.map(wd_factor).values)
    base = adj.rolling(win, center=True, min_periods=win // 2).median().interpolate(limit_direction="both")
    expected = base * v.index.dayofweek.map(wd_factor).values
    dev = v / expected - 1
    normal = dev[~excl]
    sigma = 1.4826 * float((normal - normal.median()).abs().median())  # 강건한 표준편차
    return expected, dev, sigma, wd_factor, excl


def classify(df, dev, sigma):
    hol, after = holiday_sets(df.index)
    gust2 = np.maximum(df["gust_ms"], df["gust_ms"].shift(1).fillna(df["gust_ms"]))
    snow2 = np.maximum(df["snow_cm"], df["snow_cm"].shift(1).fillna(0))
    warn = (gust2 > GUST_WARN) | (snow2 >= SNOW_WARN) | (df["precip_mm"] >= RAIN_WARN)
    weak = (gust2 >= GUST_WEAK) | (snow2 > 0)
    shock = dev <= -2 * sigma
    cause = pd.Series("", index=df.index, dtype=object)
    for d in df.index[shock]:
        if warn[d]:
            cause[d] = "W_WARN"
        elif weak[d]:
            cause[d] = "W_WEAK"
        elif d in after:
            cause[d] = "HOL"
        elif (d.month == 3 and d.day <= 20) or (d.month == 2 and d.day >= 26):
            cause[d] = "SEM"
        else:
            cause[d] = "UNK"
    return shock, cause, gust2, warn, weak, after


# ================================================================= 3. 이벤트(기상 충격 사건)
def find_events(df, dev, shock, cause):
    wdays = [d for d in df.index[shock] if cause[d] in ("W_WARN", "W_WEAK")]
    groups, cur = [], []
    for d in wdays:
        if cur and (d - cur[-1]).days > CLUSTER_GAP:
            groups.append(cur)
            cur = []
        cur.append(d)
    if cur:
        groups.append(cur)
    return [min(g, key=lambda x: dev[x]) for g in groups], groups


def path(arr, d, lo, hi, idx):
    i = idx.get_loc(d)
    if i + lo < 0 or i + hi >= len(idx):
        return None
    return arr[i + lo:i + hi + 1]


def event_stats(df, expected, dev, events, after, horizon=7, placebo_pool=None, rng=None, n_boot=5000):
    idx = df.index
    short = (expected - df["domestic"]).values
    rows = []
    for d in events:
        p = path(short, d, -1, horizon, idx)
        if p is None:
            continue
        rows.append((d, np.cumsum(p), p))
    cum = np.array([r[1] for r in rows])      # 전날(-1)부터 +horizon까지 누적 부족분
    day0 = np.array([r[2][1] for r in rows])  # 충격 당일 부족분
    post5 = np.array([r[2][2:7].sum() for r in rows])
    # 플라시보: 기상·연휴 영향이 없는 평상일을 0일로 가정해 같은 계산
    pool = placebo_pool
    pc = np.array([np.cumsum(path(short, d, -1, horizon, idx)) for d in pool])
    p5 = np.array([path(short, d, 1, 5, idx).sum() for d in pool])
    n = len(rows)
    boots_cum = np.array([pc[rng.integers(0, len(pc), n)].mean(axis=0) for _ in range(n_boot)])
    boots_p5 = np.array([p5[rng.integers(0, len(p5), n)].mean() for _ in range(n_boot)])
    lost = day0.mean()
    ratio = -(post5.mean() - p5.mean()) / lost          # 회복 비율: 0이면 회복 없음
    ratios_b = -(boots_p5 - p5.mean()) / lost
    return {
        "n": n, "lost_mean": float(lost), "lost_median": float(np.median(day0)), "lost_max": float(day0.max()),
        "recovery_ratio": float(ratio),
        "recovery_p5": float(np.percentile(ratios_b, 5)), "recovery_p95": float(np.percentile(ratios_b, 95)),
        "cum_mean": cum.mean(axis=0).tolist(),
        "band_lo": np.percentile(boots_cum, 5, axis=0).tolist(), "band_hi": np.percentile(boots_cum, 95, axis=0).tolist(),
        "days": [str(r[0].date()) for r in rows], "day0": day0.tolist(),
        "post5_mean": float(post5.mean()), "placebo_post5_mean": float(p5.mean()),
    }


# ================================================================= 4. 메인
def main():
    rng = np.random.default_rng(SEED)
    df, flights, info = load()
    M = {"data_info": info}
    dom = df["domestic"]

    # ---- 기본 통계
    M["basic"] = {
        "domestic_mean": float(dom.mean()), "domestic_median": float(dom.median()),
        "foreign_first_90d_mean": float(df["foreign"].iloc[:90].mean()),
        "foreign_last_90d_mean": float(df["foreign"].iloc[-90:].mean()),
        "foreign_share_first_year": float(df["foreign"][:"2023-12-31"].sum() / df["total"][:"2023-12-31"].sum()),
        "foreign_share_last_year": float(df["foreign"]["2025-10-01":].sum() / df["total"]["2025-10-01":].sum()),
        "gust_mean": float(df["gust_ms"].mean()), "gust_max": float(df["gust_ms"].max()),
        "gust_ge18_days": int((df["gust_ms"] >= 18).sum()), "gust_gt20_days": int((df["gust_ms"] > 20).sum()),
    }
    ma30 = df[["domestic", "foreign"]].rolling(30).mean()

    expected, dev, sigma, wd_factor, excl = build_expected(df)
    shock, cause, gust2, warn, weak, after = classify(df, dev, sigma)
    M["baseline"] = {"sigma_pct": sigma * 100, "threshold_pct": -2 * sigma * 100,
                     "n_shock": int(shock.sum()),
                     "weekday_factor": {str(k): float(v) for k, v in wd_factor.items()}}
    cnt = cause[shock].value_counts()
    M["cause_counts"] = {k: int(cnt.get(k, 0)) for k in CAUSES}
    M["largest_drops"] = [{"date": str(d.date()), "visitors": int(dom[d]), "expected": int(expected[d]),
                           "dev_pct": float(dev[d] * 100), "gust": float(df["gust_ms"][d]),
                           "precip": float(df["precip_mm"][d]), "snow": float(df["snow_cm"][d]),
                           "cause": cause[d]} for d in dev.nsmallest(8).index]

    # ---- 요일/변동성
    wk = dom.groupby(dom.index.dayofweek).mean()
    M["weekday_mean"] = {n: float(wk[i]) for i, n in enumerate(["월", "화", "수", "목", "금", "토", "일"])}
    roll_sd = (dev * 100).rolling(30).std()
    M["volatility"] = {"rolling_sd_min": float(roll_sd.min()), "rolling_sd_max": float(roll_sd.max()),
                       "rolling_sd_max_date": str(roll_sd.idxmax().date())}

    # ---- 풍속 구간별 통계 (연휴 여파 기간 제외, 당일·전날 중 큰 값)
    bins = [0, 10, 13, 16, 18, 20, 23, 99]
    labels = ["~10", "10~13", "13~16", "16~18", "18~20", "20~23", "23~"]
    mask = ~pd.Series([d in after for d in df.index], index=df.index)
    g = pd.DataFrame({"gust2": gust2, "dev": dev * 100, "shock": shock})[mask]
    g["bin"] = pd.cut(g["gust2"], bins=bins, labels=labels, right=False)
    tab = g.groupby("bin", observed=False).agg(n=("dev", "size"), mean_dev=("dev", "mean"),
                                               shock_p=("shock", "mean"))
    M["dose"] = {str(k): {"n": int(r.n), "mean_dev": float(r.mean_dev), "shock_pct": float(r.shock_p * 100)}
                 for k, r in tab.iterrows()}
    lo, hi = g[g.gust2 < 18], g[g.gust2 >= 18]
    M["dose_summary"] = {"shock_pct_lt18": float(lo.shock.mean() * 100), "shock_pct_ge18": float(hi.shock.mean() * 100),
                         "n_lt18": int(len(lo)), "n_ge18": int(len(hi)),
                         "ratio": float(hi.shock.mean() / lo.shock.mean())}
    # 단순 상관(풍속 vs 편차율)
    M["gust_corr"] = float(np.corrcoef(g["gust2"], g["dev"])[0, 1])

    # ---- 이벤트 스터디 + 회복 검증
    events, groups = find_events(df, dev, shock, cause)
    pool_mask = (~shock) & mask & (gust2 < GUST_WEAK) & (df["snow_cm"] == 0)
    all_pool = [d for d in df.index[pool_mask] if df.index.get_loc(d) >= 1 and df.index.get_loc(d) + 7 < len(df)]
    base_stats = event_stats(df, expected, dev, events, after, 7, all_pool, rng)
    ev_idx = [d for d in events if (df.index.get_loc(d) >= 1 and df.index.get_loc(d) + 7 < len(df))]
    hol_overlap = {d: any((d + pd.Timedelta(days=k)) in after for k in range(-1, 6)) for d in ev_idx}
    no_hol = [d for d in ev_idx if not hol_overlap[d]]
    no_weather_after = [d for d in ev_idx if not any(
        (df["gust_ms"][d + pd.Timedelta(days=k)] >= GUST_WEAK) or (df["snow_cm"][d + pd.Timedelta(days=k)] > 0)
        for k in range(1, 6) if (d + pd.Timedelta(days=k)) in df.index)]
    checks = {"기본(이후 5일, 평상일 비교)": base_stats}
    checks["연휴 겹친 사건 제외"] = event_stats(df, expected, dev, no_hol, after, 7, all_pool, rng)
    checks["이후 5일 기상 영향일 없는 사건"] = event_stats(df, expected, dev, no_weather_after, after, 7, all_pool, rng)
    for w_ in (21, 45):
        e2, d2, s2, _, ex2 = build_expected(df, w_)
        sh2, c2, *_ = classify(df, d2, s2)
        ev2, _ = find_events(df, d2, sh2, c2)
        pool2 = [d for d in df.index[(~sh2) & mask & (gust2 < GUST_WEAK) & (df["snow_cm"] == 0)]
                 if df.index.get_loc(d) >= 1 and df.index.get_loc(d) + 7 < len(df)]
        checks[f"기준선 창 {w_}일"] = event_stats(df, e2, d2, ev2, after, 7, pool2, rng)
    M["events"] = {k: {kk: vv for kk, vv in s.items() if kk not in ("cum_mean", "band_lo", "band_hi", "days", "day0")}
                   for k, s in checks.items()}
    M["event_days"] = base_stats["days"]
    M["event_annual_loss"] = float(np.sum(base_stats["day0"]) / (len(df) / 365.25))
    M["event_cum_mean"] = base_stats["cum_mean"]
    M["event_overlap_share"] = float(np.mean([hol_overlap[d] for d in ev_idx]))
    # 이벤트 전후 편차율 중앙값 경로
    paths = np.array([path((dev * 100).values, d, -3, 7, df.index) for d in ev_idx if path((dev * 100).values, d, -3, 7, df.index) is not None])
    M["event_dev_median"] = {str(k): float(np.median(paths[:, i])) for i, k in enumerate(range(-3, 8))}

    # ---- 항공 월별 교차검증
    mv = df["total"].resample("MS").sum().to_frame("visitors")
    fl = flights.copy()
    fl["month"] = pd.to_datetime(fl["ym"].astype(str), format="%Y%m")
    j = mv.merge(fl.set_index("month"), left_index=True, right_index=True, how="inner")
    j["mom_v"] = j["visitors"].pct_change()
    j["mom_p"] = j["passengers"].pct_change()
    j.loc[j.index.to_series().diff().dt.days > 35, ["mom_v", "mom_p"]] = np.nan  # 누락 월 뒤는 제외
    M["flights"] = {"months": int(len(j)), "corr_level": float(j["visitors"].corr(j["passengers"])),
                    "corr_mom": float(j[["mom_v", "mom_p"]].dropna().corr().iloc[0, 1]),
                    "missing_month": sorted(set(pd.date_range("2023-01-01", "2026-07-01", freq="MS").strftime("%Y%m"))
                                            - set(fl["ym"].astype(str)))}
    # 월별 충격 건수와 운항횟수
    ev_m = pd.Series(1, index=pd.DatetimeIndex(events)).resample("MS").sum().reindex(j.index).fillna(0)
    j["events"] = ev_m
    j["flights_per_day"] = j["flights"] / j.index.days_in_month
    M["flights"]["corr_events_flights"] = float(j["events"].corr(j["flights_per_day"]))

    # ---- 시계열 분해 (추세 + 연간 계절성 + 요일 + 잔차)
    trend = dom.rolling(365, center=True, min_periods=300).mean()
    detr = dom - trend
    seas_m = detr.groupby(detr.index.month).transform("mean")
    seas_w = (detr - seas_m).groupby(detr.index.dayofweek).transform("mean")
    resid = dom - trend - seas_m - seas_w
    rs = float(resid[~shock].std())  # 충격일을 뺀 잔차의 표준편차
    M["decomp"] = {"trend_start": float(trend.dropna().iloc[0]), "trend_end": float(trend.dropna().iloc[-1]),
                   "trend_change_pct": float((trend.dropna().iloc[-1] / trend.dropna().iloc[0] - 1) * 100),
                   "season_peak_month": int(seas_m.groupby(seas_m.index.month).first().idxmax()),
                   "season_low_month": int(seas_m.groupby(seas_m.index.month).first().idxmin()),
                   "season_range": float(seas_m.max() - seas_m.min()), "resid_std": float(rs),
                   "shock_resid_mean": float(resid[shock].mean()), "shock_resid_below_2sd_pct": float((resid[shock] < -2 * rs).mean() * 100),
                   "normal_resid_below_2sd_pct": float((resid[~shock] < -2 * rs).mean() * 100)}

    # ---- 베이스라인 예측 (마지막 28일 보류)
    H = 28
    train, test = dom.iloc[:-H], dom.iloc[-H:]
    lvl = train.iloc[-28:].mean() / train.iloc[-28 - 364:-364].mean()
    snaive = pd.Series([train.get(d - pd.Timedelta(days=364)) for d in test.index], index=test.index) * lvl
    wk_mean = train.iloc[-28:].groupby(train.iloc[-28:].index.dayofweek).mean()
    naive = pd.Series(test.index.dayofweek.map(wk_mean).values, index=test.index)
    mape = lambda a, b: float((np.abs(a - b) / a).mean() * 100)
    M["forecast"] = {"horizon_days": H, "mape_seasonal": mape(test, snaive), "mape_weekday_mean": mape(test, naive),
                     "level_ratio": float(lvl), "test_start": str(test.index[0].date()), "test_end": str(test.index[-1].date())}

    # ================= 시각화 =================
    def money(x, _):
        return f"{int(x):,}"

    # 01 추세
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.plot(dom.index, dom, color=BLUE, alpha=.18, lw=.7)
    ax.plot(df["foreign"].index, df["foreign"], color=ORANGE, alpha=.18, lw=.7)
    ax.plot(ma30.index, ma30["domestic"], color=BLUE, lw=2.2, label="내국인 (30일 이동평균)")
    ax.plot(ma30.index, ma30["foreign"], color=ORANGE, lw=2.2, label="외국인 (30일 이동평균)")
    ax.set_title("일별 입도 관광객과 30일 이동평균 (2023.1~2026.9)", loc="left", fontweight="bold")
    ax.set_ylabel("명/일"); ax.yaxis.set_major_formatter(plt.FuncFormatter(money))
    ax.legend(loc="center right", frameon=False)
    fig.text(0.01, -0.02, "출처: 지자체 관광빅데이터플랫폼 '입도 관광객 추이(일별)'. 흐린 선 = 일별 원값", fontsize=8, color=GRAY)
    fig.savefig(IMG / "01_trend_moving_average.png"); plt.close(fig)

    # 02 편차율·충격일
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.axhspan(-2 * sigma * 100, 2 * sigma * 100, color="#EFEDE9", zorder=0)
    ax.plot(dev.index, dev * 100, color="#A7A59B", lw=.8)
    for c, name in CAUSES.items():
        s = dev[shock & (cause == c)]
        ax.scatter(s.index, s * 100, s=22, color=CAUSE_COLOR[c], edgecolor="white", lw=.6, zorder=3,
                   label=f"{name} {len(s)}일")
    ax.set_ylim(-100, 75); ax.set_ylabel("편차율 (%)")
    ax.set_title("내국인 입도객의 '평소 대비 편차율'과 충격일 (요일 보정 31일 이동중앙값 기준)", loc="left", fontweight="bold")
    ax.legend(ncol=3, frameon=False, loc="upper center", fontsize=8.5)
    fig.savefig(IMG / "02_deviation_shock_days.png"); plt.close(fig)

    # 03 풍속 용량-반응
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    cols = [BLUE if i < 4 else ORANGE for i in range(7)]
    xl3 = [f"{l}\n(n={n})" for l, n in zip(labels, tab.n)]
    a = axs[0]; b = a.bar(xl3, tab.mean_dev, color=cols, width=.62)
    for r, vv in zip(b, tab.mean_dev):
        a.annotate(f"{vv:+.1f}%", (r.get_x() + r.get_width() / 2, vv), ha="center",
                   va="bottom" if vv >= 0 else "top", fontsize=9, xytext=(0, 3 if vv >= 0 else -3), textcoords="offset points")
    a.axhline(0, color="#555", lw=.8); a.set_title("평균 편차율", loc="left", fontweight="bold")
    a.set_ylabel("내국인 입도객 편차율 (%)"); a.set_xlabel("최대 순간풍속 (m/s, 당일·전날 중 큰 값)")
    a = axs[1]; b = a.bar(xl3, tab.shock_p * 100, color=cols, width=.62)
    for r, vv in zip(b, tab.shock_p * 100):
        a.annotate(f"{vv:.0f}%", (r.get_x() + r.get_width() / 2, vv), ha="center", va="bottom", fontsize=9,
                   xytext=(0, 3), textcoords="offset points")
    a.set_title("충격일(편차율 -2σ 이하) 발생 확률", loc="left", fontweight="bold")
    a.set_ylabel("%"); a.set_xlabel("최대 순간풍속 (m/s)")
    ds = M["dose_summary"]
    fig.suptitle(f"순간풍속 18m/s 이상에서 충격일 확률 {ds['shock_pct_lt18']:.0f}% → {ds['shock_pct_ge18']:.0f}% (약 {ds['ratio']:.1f}배)",
                 x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.text(0.01, -0.07, "연휴·명절 여파 기간 제외. 기상청 강풍주의보 기준은 순간풍속 20m/s 초과. n = 해당 구간 일수", fontsize=8, color=GRAY)
    fig.subplots_adjust(top=0.80, wspace=0.22)
    fig.savefig(IMG / "03_gust_dose_response.png"); plt.close(fig)

    # 04 이벤트 스터디
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axhspan(-2 * sigma * 100, 2 * sigma * 100, color="#EFEDE9", zorder=0)
    xs = np.arange(-3, 8)
    for p in paths:
        ax.plot(xs, p, color=BLUE, alpha=.15, lw=1)
    med = np.median(paths, axis=0)
    ax.plot(xs, med, color=BLUE, lw=3, marker="o", label=f"중앙값 경로 ({len(paths)}개 사건)")
    for x, y in zip(xs, med):
        if x in (-1, 0, 1, 2):
            ax.annotate(f"{y:.0f}%", (x, y), xytext=(8, -12 if x == 0 else 2), textcoords="offset points", fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels([("충격\n당일" if x == 0 else f"{x:+d}일") for x in xs])
    ax.set_ylabel("내국인 입도객 편차율 (%)"); ax.set_ylim(-100, 32)
    ax.set_title(f"기상 충격 전후 7일: 당일 {med[3]:.0f}% 급감 후 다음 날 {med[4]:.0f}% 수준으로 복귀", loc="left", fontweight="bold")
    ax.legend(loc="lower right", frameon=False)
    fig.text(0.01, -0.02, "흐린 선 = 개별 사건. 기상 원인 충격일이 2일 이내 간격이면 하나로 묶고, 편차율이 가장 낮은 날을 0일로 정렬", fontsize=8, color=GRAY)
    fig.savefig(IMG / "04_event_study.png"); plt.close(fig)

    # 05 누적 순감소
    cm = np.array(base_stats["cum_mean"]); lo_, hi_ = np.array(base_stats["band_lo"]), np.array(base_stats["band_hi"])
    fig, ax = plt.subplots(figsize=(11, 4.8))
    xl = ["전날", "충격\n당일"] + [f"+{k}일" for k in range(1, 8)]
    ax.fill_between(range(len(cm)), lo_, hi_, color="#E9E7E1", label=f"기상 영향 없던 평상일 {base_stats['n']}개로 같은 계산을 했을 때 (90% 범위)")
    ax.plot(range(len(cm)), cm, color=ORANGE, lw=3.2, marker="o", label=f"기상 충격 {base_stats['n']}건 평균")
    ax.axhline(0, color="#555", lw=.8)
    ax.annotate(f"{cm[1]:,.0f}명", (1, cm[1]), xytext=(0, 9), textcoords="offset points", ha="center")
    ax.annotate(f"{cm[-1]:,.0f}명", (len(cm) - 1, cm[-1]), xytext=(0, 9), textcoords="offset points", ha="center")
    ax.text(4, 9300, f"충격 당일 이후 7일 동안 누적 부족분 변화 {cm[-1] - cm[1]:+,.0f}명\n→ 평상일 범위 안, 다시 들어온 흔적 없음",
            ha="center", va="center", fontsize=10, color="#555")
    ax.set_xticks(range(len(cm))); ax.set_xticklabels(xl)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(money)); ax.set_ylabel("누적으로 들어오지 못한 관광객 (명, 사건 평균)")
    ax.set_title("결항으로 빠진 관광객은 일주일이 지나도 메워지지 않았다", loc="left", fontweight="bold")
    ax.legend(loc="center right", frameon=False, fontsize=9)
    fig.text(0.01, -0.03, "내국인 입도객의 '평소 기대치 대비 부족분'을 전날부터 누적. 늦게라도 들어왔다면 선이 0 쪽으로 내려와야 함.", fontsize=8, color=GRAY)
    fig.savefig(IMG / "05_cumulative_loss.png"); plt.close(fig)

    # 06 요일·변동성
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw={"width_ratios": [1, 1.5]})
    a = axs[0]; b = a.bar(list(M["weekday_mean"].keys()), list(M["weekday_mean"].values()), color=BLUE, width=.6)
    for r, vv in zip(b, M["weekday_mean"].values()):
        a.annotate(f"{vv/1000:.1f}천", (r.get_x() + r.get_width() / 2, vv), ha="center", va="bottom", fontsize=9,
                   xytext=(0, 3), textcoords="offset points")
    a.set_ylim(25000, 36500); a.set_title("요일별 평균 내국인 입도객", loc="left", fontweight="bold")
    a.yaxis.set_major_formatter(plt.FuncFormatter(money))
    a = axs[1]; a.plot(roll_sd.index, roll_sd, color=BLUE, lw=1.8)
    a.set_title("편차율의 30일 롤링 표준편차 (변동성)", loc="left", fontweight="bold"); a.set_ylabel("%p")
    fig.text(0.01, -0.03, "왼쪽 막대는 세로축이 25,000에서 시작함(요일 간 차이를 보기 위함)", fontsize=8, color=GRAY)
    fig.savefig(IMG / "06_weekday_volatility.png"); plt.close(fig)

    # 07 분해
    fig, axs = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    axs[0].plot(dom.index, dom, color=BLUE, lw=.7); axs[0].set_title("원 데이터 (내국인)", loc="left", fontsize=10, fontweight="bold")
    axs[1].plot(trend.index, trend, color=ORANGE, lw=2); axs[1].set_title("추세 (365일 중심 이동평균)", loc="left", fontsize=10, fontweight="bold")
    axs[2].plot((seas_m + seas_w).index, seas_m + seas_w, color=GREEN, lw=.8); axs[2].set_title("계절성 (월 평균 패턴 + 요일 패턴)", loc="left", fontsize=10, fontweight="bold")
    axs[3].plot(resid.index, resid, color=GRAY, lw=.7)
    axs[3].scatter(resid[shock].index, resid[shock], s=10, color=ORANGE, zorder=3, label="충격일")
    axs[3].axhline(-2 * rs, color="#C44", lw=.8, ls="--"); axs[3].legend(frameon=False, loc="lower right")
    axs[3].set_title("잔차 (점선 = 충격일을 뺀 잔차의 -2 표준편차)", loc="left", fontsize=10, fontweight="bold")
    for a in axs:
        a.yaxis.set_major_formatter(plt.FuncFormatter(money))
    fig.suptitle("시계열 분해: 추세 · 계절성 · 잔차", x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.savefig(IMG / "07_decomposition.png"); plt.close(fig)

    # 08 예측
    fig, ax = plt.subplots(figsize=(11, 4.4))
    hist = dom.iloc[-90:-H]
    ax.plot(hist.index, hist, color="#9CB9E0", lw=1.5, label="학습 구간 실제값")
    ax.plot(test.index, test, color=BLUE, lw=2.4, marker="o", ms=3.5, label="실제값 (보류 28일)")
    ax.plot(snaive.index, snaive, color=ORANGE, lw=2, ls="--", label=f"전년 같은 요일 × 최근 수준비 (MAPE {M['forecast']['mape_seasonal']:.1f}%)")
    ax.plot(naive.index, naive, color=GREEN, lw=2, ls=":", label=f"최근 4주 요일 평균 (MAPE {M['forecast']['mape_weekday_mean']:.1f}%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(money))
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    ax.set_title("베이스라인 예측: 마지막 28일을 가리고 맞혀 보기", loc="left", fontweight="bold")
    fig.text(0.01, -0.14, "기상·연휴를 반영하지 않은 단순 기준선이라 정확도보다 '기준선이 얼마나 어긋나는지'를 보는 용도", fontsize=8, color=GRAY)
    fig.savefig(IMG / "08_forecast_baseline.png"); plt.close(fig)

    # 09 항공 vs 입도객
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    a = axs[0]; a.scatter(j["passengers"] / 1e6, j["visitors"] / 1e3, color=BLUE, s=36)
    for d_, r in j.iterrows():
        if d_.month in (1, 2) and d_.year == 2025:
            a.annotate(d_.strftime("%y.%m"), (r.passengers / 1e6, r.visitors / 1e3), fontsize=8, xytext=(4, 4), textcoords="offset points")
    a.set_xlabel("월 항공 여객 (백만 명)"); a.set_ylabel("월 입도 관광객 (천 명)")
    a.set_title(f"월 단위 여객과 입도객 (상관계수 {M['flights']['corr_level']:.2f})", loc="left", fontweight="bold")
    a = axs[1]; a.plot(j.index, j["visitors"] / j["visitors"].mean(), color=BLUE, lw=2, label="월 입도객")
    a.plot(j.index, j["passengers"] / j["passengers"].mean(), color=ORANGE, lw=2, label="월 항공 여객")
    a.set_title("월별 추이 (각 평균 = 1)", loc="left", fontweight="bold"); a.legend(frameon=False)
    fig.text(0.01, -0.03, f"항공 자료에 없는 달({', '.join(M['flights']['missing_month'])})은 제외", fontsize=8, color=GRAY)
    fig.savefig(IMG / "09_flights_vs_visitors.png"); plt.close(fig)

    # ================= 대시보드 데이터 =================
    code = {"": 0, "W_WARN": 1, "W_WEAK": 2, "HOL": 3, "SEM": 4, "UNK": 5}
    dd = {"dates": [d.strftime("%Y-%m-%d") for d in df.index], "visitors": dom.astype(int).tolist(),
          "expected": expected.round(0).astype(int).tolist(), "dev": (dev * 100).round(1).tolist(),
          "gust": df["gust_ms"].tolist(), "gust2": gust2.round(1).tolist(), "precip": df["precip_mm"].tolist(),
          "snow": df["snow_cm"].tolist(), "shock": [int(x) for x in shock], "cause": [code[c] for c in cause],
          "sigma": round(sigma * 100, 2), "dow": [int(x) for x in df.index.dayofweek],
          "month": [int(x) for x in df.index.month], "in_after": [int(d in after) for d in df.index]}
    (DASH / "data.js").write_text("window.DATA = " + json.dumps(dd, ensure_ascii=False) + ";\n", encoding="utf-8")

    # 요약 저장
    def cvt(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError
    (RES / "metrics.json").write_text(json.dumps(M, ensure_ascii=False, indent=2, default=cvt), encoding="utf-8")
    print(json.dumps(M, ensure_ascii=False, indent=1, default=cvt)[:6000])


if __name__ == "__main__":
    main()
