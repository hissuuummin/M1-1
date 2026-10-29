# 기상 조건이 섬 관광지 입도 관광객에 미치는 영향 분석

일별 입도 관광객(2023.1~2026.9)과 기상 일자료를 겹쳐서 **"어떤 날씨부터 관광객이 급감하는가, 줄어든 관광객은 돌아오는가"** 를 시계열로 분석한 과제입니다.

- 분석 리포트: **[REPORT.md](REPORT.md)**
- 핵심 결과: 순간풍속 18m/s 이상에서 입도객 급감(평소 대비 −2σ 이하) 확률이 약 5% → 25%로 5배. 급감 하루 뒤 일별 수치는 정상화되지만 빠진 인원이 돌아왔다는 근거는 없음(5가지 검증).
- 보너스: 대시보드(`dashboard/index.html`), 시계열 분해·베이스라인 예측

## 폴더 구조

```
.
|-- data/
|   |-- visitors_daily.csv    # 일별 입도 관광객 (내국인/외국인/전체), 1,369일
|   |-- weather_daily.csv     # 일강수량, 최대 순간풍속, 최대 풍속, 일 최심신적설
|   |-- flights_monthly.csv   # 월별 항공 운항·여객 (2025-12 없음)
|-- images/                   # 분석 그래프 01~09, dashboard/ 스크린샷
|-- dashboard/                # 브라우저로 여는 대시보드 (index.html + data.js)
|-- results/metrics.json      # 리포트에 쓴 모든 수치
|-- analysis.py               # 분석 + 그래프 + 대시보드 데이터 생성
|-- prepare_data.py           # 원본 CSV → data/ 정리
|-- REPORT.md
|-- requirements.txt
```

## 실행 방법

Python 3.10 이상 (3.13, pandas 3.0, matplotlib 3.11에서 테스트).

```bash
pip install -r requirements.txt
python analysis.py
```

실행하면 `images/01~09_*.png`, `results/metrics.json`, `dashboard/data.js`가 새로 만들어집니다.
그래프의 한글은 Noto Sans CJK / NanumGothic / Malgun Gothic / AppleGothic 중 설치된 폰트를 자동으로 씁니다. 없으면 글자가 깨질 수 있습니다.

### 대시보드

`dashboard/index.html`을 브라우저로 열면 됩니다(서버 불필요, 같은 폴더의 `data.js` 필요).
`python analysis.py`를 다시 실행하면 `data.js`가 갱신됩니다.

### 원본 데이터에서 다시 만들기 (선택)

```bash
python prepare_data.py --visitors <입도 관광객 추이(일별).csv> --weather <기상청 ASOS 일자료.csv> --flights <월별 항공 운항 및 여객 추이.csv>
```

기상 파일은 cp949 인코딩이고 관측소가 여러 곳 섞여 있어서, 지점번호 하나만 남겨 `data/weather_daily.csv`로 저장합니다.

## 데이터 출처 및 주의

| 데이터 | 출처 |
|---|---|
| 입도 관광객 추이(일별), 항공 운항·여객(월별) | 지자체 관광빅데이터플랫폼 |
| 기상 일자료 (종관기상관측, ASOS) | 기상청 기상자료개방포털 |

- 기간: 2023-01-01 ~ 2026-09-30 (항공 자료는 월별, 2023-01 ~ 2026-07)
- 공공데이터 이용 조건을 따릅니다. 원자료를 재배포·상업적으로 사용할 때는 각 제공처의 이용 조건을 확인하세요.
- 공휴일·연휴 표는 `analysis.py`의 `HOLIDAY_BLOCKS`에 직접 정리한 값이며 임시공휴일이 일부 빠졌을 수 있습니다.
