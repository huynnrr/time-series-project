# Forecasting International Tourist Arrivals to Vietnam using SARIMA/ARIMAX
### The Role of Online Search Behavior

> **Time Series Analysis Project**  
> Forecasting monthly international tourist arrivals to Vietnam (2010–2023) using econometric and machine learning models, with Google Trends as an exogenous variable.

---

## Research Question

> *Does incorporating Google Search Trends improve forecasting performance for international tourist arrivals to Vietnam, particularly during structural breaks such as COVID-19?*

---

## Project Structure

```
vietnam-tourism-arimax/
│
├── dataset/
│   ├── vietnam_tourism_trends_monthly_2010_2023.csv   # Raw data
│   └── processed_data.csv                             # Feature-engineered data
│
├── outputs/                          # All plots and metrics (.png, .csv)
│
├── EDA_exploration.ipynb             # Bước 1 — Exploratory Data Analysis
├── Feature_engineering.ipynb        # Bước 2 — Feature Engineering
├── Stationarity_tests.ipynb         # Bước 3 — ADF, KPSS, ACF/PACF
├── SARIMA_model.ipynb               # Bước 4 — SARIMA baseline
├── ARIMAX_model.ipynb               # Bước 5 — ARIMAX (main model)
├── ML_models.ipynb                  # Bước 6 — Random Forest & XGBoost
├── Evaluation.ipynb                 # Bước 7 — Model comparison
├── Rolling_Forecast.ipynb           # Bước 8 — Walk-forward rolling forecast
│
├── requirements.txt
└── README.md
```

---

## Data

| Variable | Description | Source |
|---|---|---|
| `arrivals` | Monthly international tourist arrivals to Vietnam | Vietnam National Authority of Tourism |
| `search_trends` | Google Trends index for "Vietnam tourism" (0–100) | Google Trends |

- **Period**: January 2010 – December 2023 (168 observations)
- **Frequency**: Monthly
- **Train set**: 2010–2019 (120 obs) — pre-COVID only
- **Test set**: 2020–2023 (48 obs) — includes COVID shock + recovery

---

## Models

| Model | Type | Exogenous |
|---|---|---|
| SARIMA(0,1,1)(0,1,1)[12] | Econometric | None |
| ARIMAX(0,1,1)(0,1,1)[12] | Econometric | `log_search_trends` |
| Random Forest | Machine Learning | lag features + search_trends + covid_dummy |
| XGBoost | Machine Learning | lag features + search_trends + covid_dummy |

---

## Key Results

### Static Forecast (RMSE, log scale)

| Model | Overall | COVID | Recovery |
|---|---|---|---|
| SARIMA | 9.397 | 13.689 | 2.203 |
| ARIMAX | 9.278 | 13.513 | 2.183 |
| Random Forest | **8.431** ★ | **12.402** | **1.082** |
| XGBoost | 8.465 | 12.450 | 1.106 |

### Walk-forward Rolling Forecast (RMSE, log scale)

| Model | Static | Rolling | Improvement |
|---|---|---|---|
| SARIMA | 9.40 | 2.67 | 71.6% |
| ARIMAX | 9.28 | **2.24** ★ | 75.9% |
| Random Forest | 8.43 | 3.29 | 61.0% |
| XGBoost | 8.46 | 2.94 | 65.2% |

### Key Findings

1. **ML wins on static RMSE** — Random Forest best overall, driven by lag/rolling features
2. **ARIMAX wins on rolling forecast** — search_trends adds value when updated monthly
3. **ARIMAX > SARIMA** — ΔAIC = −0.076 (marginal), but `log_search_trends` coefficient = 0.40 (p = 0.013) and rolling RMSE improvement is substantial
4. **COVID = structural break** — all models struggle in Mar–Apr 2020; rolling forecast recovers faster
5. **Search trends as coincident indicator** — Granger causality not significant in normal periods
(r = −0.207 pre-COVID on log scale), but r = 0.89 across full series — driven by COVID collapse

---

## Installation

```bash
# Clone repository
git clone https://github.com/your-username/vietnam-tourism-arimax.git
cd vietnam-tourism-arimax

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

---

## Run Order

Run notebooks in this order:

```
1. EDA_exploration.ipynb
2. Feature_engineering.ipynb       ← generates processed_data.csv
3. Stationarity_tests.ipynb
4. SARIMA_model.ipynb
5. ARIMAX_model.ipynb
6. ML_models.ipynb
7. Evaluation.ipynb
8. Rolling_Forecast.ipynb
```

> **Note**: `Feature_engineering.ipynb` must be run before any model notebooks — it generates `dataset/processed_data.csv` used by all subsequent steps.

---

## Limitations

- `log_search_trends` in ARIMAX uses **actual values** from the test period — in real deployment, search trends would need to be forecasted or sourced via nowcasting
- `covid_dummy` cannot be estimated in ARIMAX training (all zeros in 2010–2019) — its effect is partially captured through `log_search_trends`
- ML rolling forecast performance degrades relative to econometric models — tree-based models require sufficient stable data for each re-fit
- Heteroskedasticity detected in SARIMA/ARIMAX residuals (tourism data has growing variance over time)

---

## Dependencies

```
pandas          2.2.2
numpy           1.26.4
matplotlib      3.8.4
statsmodels     0.14.2
scikit-learn    1.4.2
xgboost         2.0.3
pmdarima        2.0.4
scipy           1.13.0
```
