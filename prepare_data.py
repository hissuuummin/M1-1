"""원본 CSV 3개를 분석용 data/ 파일로 정리한다.

사용법:
    python prepare_data.py --visitors <입도객 일별 csv> --weather <기상 일자료 csv> --flights <항공 월별 csv>

- 기상 파일은 여러 관측소가 섞여 있어서 지점번호 STATION_ID 한 곳만 남기고 지점 이름 열은 버린다.
- 기상 파일은 cp949 인코딩이다.
"""
import argparse
from pathlib import Path

import pandas as pd

STATION_ID = 184  # 분석에 쓰는 섬 중심 관측소 1곳의 지점번호
OUT = Path(__file__).parent / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--visitors", required=True)
    ap.add_argument("--weather", required=True)
    ap.add_argument("--flights", required=True)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    v = pd.read_csv(a.visitors, encoding="utf-8-sig")
    v.columns = ["date", "domestic", "foreign", "total"]
    v["date"] = pd.to_datetime(v["date"].astype(str), format="%Y%m%d")
    v = v.sort_values("date").reset_index(drop=True)
    assert v["date"].is_unique and len(v) >= 100
    v.to_csv(OUT / "visitors_daily.csv", index=False, date_format="%Y-%m-%d")

    w = pd.read_csv(a.weather, encoding="cp949")
    w.columns = ["station", "station_name", "date", "precip_mm", "gust_ms", "wind_ms", "snow_cm"]
    w = w[w["station"] == STATION_ID].copy()
    w["date"] = pd.to_datetime(w["date"])
    # 강수·적설 빈 칸은 '해당 없음'(비·눈이 오지 않은 날)이므로 0으로 채운다
    w[["precip_mm", "snow_cm"]] = w[["precip_mm", "snow_cm"]].fillna(0.0)
    w = w[["date", "precip_mm", "gust_ms", "wind_ms", "snow_cm"]].sort_values("date")
    w.to_csv(OUT / "weather_daily.csv", index=False, date_format="%Y-%m-%d")

    f = pd.read_csv(a.flights, encoding="utf-8-sig")
    f.columns = ["ym", "flights", "passengers"]
    f = f.sort_values("ym")
    f.to_csv(OUT / "flights_monthly.csv", index=False)

    m = v.merge(w, on="date", how="left")
    print("visitors", len(v), v.date.min().date(), v.date.max().date())
    print("weather ", len(w), "unmatched:", int(m["gust_ms"].isna().sum()))
    print("flights ", len(f), f.ym.min(), f.ym.max())


if __name__ == "__main__":
    main()
