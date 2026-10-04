# Calgary Traffic Risk Intelligence

A data-science and machine-learning portfolio project that analyzes **recorded Calgary traffic incidents** using temporal patterns, geographic clustering, traffic-control locations, historical weather, and chronological predictive modelling.

## Project Questions

The project explores:

- Where are recurring geographic concentrations of recorded traffic incidents?
- Which hours, weekdays, months, and seasons have the most recorded incidents?
- How do hotspot patterns vary across time?
- Which traffic-control locations have the highest historical incident activity?
- How does recorded incident frequency differ during snow, rain, freezing temperatures, low visibility, high wind, and other project-defined weather conditions?
- Which factors contribute most to the project-defined Historical Incident Risk Index?
- Can time and weather features improve hourly citywide incident-activity estimates over a simple historical baseline?

## Important Source Caveat

The City of Calgary describes the Traffic Incidents dataset as an **unofficial archive of traffic disruptions** and notes that gaps can occur because of system or script malfunction.

For weather-rate analysis and predictive modelling, this project conservatively flags intervals where more than 48 hours pass between consecutive recorded incidents. Hours inside those inferred gaps are excluded from exposure-rate calculations and model evaluation instead of being treated as genuine zero-incident hours.

The 48-hour rule is a project heuristic, not an official City uptime indicator.

## Project Pipeline

```text
City of Calgary Traffic Incidents
              +
City of Calgary Traffic Signals
              |
              v
       Data Preparation
              |
              v
  Exploratory Data Analysis
              |
              v
    Haversine DBSCAN Hotspots
              |
              v
 Nearest Traffic-Control Matching
              |
              +----------------------+
              |                      |
              v                      v
        ECCC Weather          Spatial Features
              |                      |
              +----------+-----------+
                         |
                         v
            Weather / Risk Analysis
       + source-gap-aware hourly exposure
                         |
                         v
              Predictive Modelling
                         |
                         v
              Streamlit Dashboard
```

## Notebooks

### `01_data_preparation.ipynb`

- loads the two raw City of Calgary snapshots
- checks source quality, missing values, duplicates, identifiers, and coordinates
- parses UTC incident timestamps
- converts incident times to `America/Edmonton`
- engineers year/month/weekday/hour/season/time-period features
- cleans traffic-control records and installation dates
- exports `incidents_clean.csv` and `signals_clean.csv`

### `02_exploratory_analysis.ipynb`

- yearly, monthly, weekday, hourly, seasonal, and time-period patterns
- source incident descriptions and quadrants
- coordinate sanity checks
- incident and traffic-control map
- explicit warning that archive counts are not exposure-normalized risk estimates

The snapshot starts in December 2016 and ends in September 2026, and the source also contains internal collection gaps. Annual values are therefore described as **recorded archive counts**, not complete exposure-normalized annual rates.

### `03_hotspot_and_intersection_analysis.ipynb`

- Haversine DBSCAN hotspot detection
- DBSCAN sensitivity checks
- hotspot summaries
- BallTree nearest-neighbour search
- traffic-control matching sensitivity checks
- thresholded traffic-control matching

Project parameters:

- DBSCAN `eps = 50 m`
- DBSCAN `min_samples = 20`
- traffic-control matching threshold `75 m`

### `04_weather_and_risk_analysis.ipynb`

- downloads and caches ECCC hourly weather month-by-month
- converts fixed Calgary LST weather timestamps to UTC
- joins weather to incidents by UTC hour
- preserves missing weather measurements as unknown where appropriate
- creates an hourly exposure dataset including zero-incident hours
- detects obvious source-collection gaps
- performs exposure-adjusted weather comparisons
- summarizes incident frequency by temperature band
- creates time-specific hotspot profiles
- creates the Historical Incident Risk Index
- uses a coverage-adjusted trend component
- exports compact dashboard-specific files so deployment does not require the largest intermediate CSVs

The hourly precipitation field in the downloaded ECCC snapshot is not sufficiently populated for predictive use, so precipitation is **not** used as a model feature.

### `05_predictive_modeling.ipynb`

- uses eligible observed hours from 2017–2024 for training
- uses eligible observed hours from 2025 for chronological testing
- compares against a historical weekday/hour baseline
- uses cyclical time features and weather features
- uses median imputation with missing-value indicators
- trains a Poisson `HistGradientBoostingRegressor`
- evaluates MAE, RMSE, R², and Mean Poisson Deviance
- calculates permutation feature importance on held-out data
- saves the trained model, metadata, metrics, importance, and test predictions

Final model claims should use the **new metrics produced after rerunning Notebook 05**. If the ML model only slightly improves on, or fails to beat, the baseline, that result should be reported as-is.

## Historical Incident Risk Index

The project creates a transparent 0–100 analytical index for traffic-control locations using the 2017–2025 analysis window.

| Component | Weight |
|---|---:|
| Historical matched incident count | 30% |
| Recent incident activity (2024–2025) | 25% |
| DBSCAN hotspot concentration | 15% |
| Adverse-weather incident share | 10% |
| Rush-hour concentration | 10% |
| Positive coverage-adjusted historical trend | 10% |

This is a **project-defined historical incident indicator**. It is not an official City of Calgary road-safety rating and does not estimate collision probability.

## Technology

Python, Pandas, NumPy, Scikit-learn, Plotly, Streamlit, Requests, Joblib, Jupyter, DBSCAN, BallTree/Haversine distance, and gradient boosting with Poisson loss.

## Data Sources

### City of Calgary — Traffic Incidents

https://data.calgary.ca/Transportation-Transit/Traffic-Incidents/35ra-9556

Local project snapshot: `Traffic_Incidents_20260928.csv`

### City of Calgary — Traffic Signals

https://data.calgary.ca/Health-and-Safety/Traffic-Signals/qr97-4jvx

Local project snapshot: `Traffic_Signals_20260928.csv`

### Environment and Climate Change Canada — Historical Climate Data

https://climate.weather.gc.ca/

Weather station used:

- `CALGARY INTL A`
- Climate ID `3031092`
- ECCC station ID `50430`
- Airport `YYC`

Review and follow the applicable source-data licences when redistributing source or derived data.

## Folder Structure

```text
calgary-traffic-risk-intelligence/
│
├── .streamlit/
│   └── config.toml
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/                         # local only; Git-ignored
│   │   ├── weather/                 # ECCC monthly cache; Git-ignored
│   │   ├── Traffic_Incidents_20260928.csv
│   │   └── Traffic_Signals_20260928.csv
│   │
│   └── cleaned/                     # generated by notebooks
│       ├── dashboard_incidents.csv              # deploy
│       ├── hotspot_summary.csv                  # deploy
│       ├── hotspot_time_profile.csv             # deploy
│       ├── traffic_control_risk_summary.csv     # deploy
│       ├── weather_condition_comparison.csv     # deploy
│       ├── temperature_band_summary.csv         # deploy
│       ├── model_metrics.csv                    # deploy
│       ├── model_feature_importance.csv         # deploy
│       ├── model_test_predictions_2025.csv      # deploy
│       └── other analysis/intermediate CSVs     # local only
│
├── models/
│   ├── hourly_incident_model.joblib
│   └── hourly_incident_model_metadata.json
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_hotspot_and_intersection_analysis.ipynb
│   ├── 04_weather_and_risk_analysis.ipynb
│   └── 05_predictive_modeling.ipynb
│
├── scripts/
│   └── reset_generated_outputs.ps1
│
├── DEPLOYMENT.md
├── FINAL_CHECKLIST.md
├── README.md
├── requirements.txt
└── .gitignore
```

## Clean Rebuild

Keep `data/raw/` and the cached `data/raw/weather/` folder. Remove old generated outputs from `data/cleaned/` and `models/`, then rerun the notebooks in order.

On Windows PowerShell you can use the included helper:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/reset_generated_outputs.ps1
```

Then activate the environment and install dependencies:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run:

```text
01_data_preparation.ipynb
02_exploratory_analysis.ipynb
03_hotspot_and_intersection_analysis.ipynb
04_weather_and_risk_analysis.ipynb
05_predictive_modeling.ipynb
```

Then launch the dashboard from the project root:

```powershell
python -m streamlit run app/app.py
```

## Files Required by the Deployed Dashboard

```text
data/cleaned/dashboard_incidents.csv
data/cleaned/hotspot_summary.csv
data/cleaned/hotspot_time_profile.csv
data/cleaned/traffic_control_risk_summary.csv
data/cleaned/weather_condition_comparison.csv
data/cleaned/temperature_band_summary.csv
data/cleaned/model_metrics.csv
data/cleaned/model_feature_importance.csv
data/cleaned/model_test_predictions_2025.csv

models/hourly_incident_model.joblib
models/hourly_incident_model_metadata.json
```

The larger analysis files such as `incidents_weather.csv`, `hourly_incident_weather.csv`, and `weather_hourly_calgary.csv` are regenerated locally and are not required by the deployed Streamlit application.

## Git Strategy

The included `.gitignore` is configured to:

- ignore `.venv/`, caches, editor files, and Streamlit secrets
- ignore the raw City snapshots and ECCC monthly weather cache
- ignore intermediate generated CSVs by default
- keep only the compact generated CSVs required by the deployed app
- keep the two trained model files in `models/`

Before the first Git commit, rerun all notebooks and verify the final generated file sizes.

## Limitations

- recorded traffic incidents are not direct collision probabilities
- the City source is an unofficial archive and can contain collection gaps
- the >48-hour collection-gap rule is a project heuristic
- traffic-volume exposure is not included
- road length/exposure is not normalized
- Calgary airport weather is used as a citywide weather proxy
- weather relationships are observational and do not establish causation
- DBSCAN results depend on selected clustering parameters
- traffic-control matching depends on the 75 m threshold
- the Historical Incident Risk Index is project-defined
- the predictive model estimates citywide hourly recorded incident activity, not individual crashes
- the application is not a real-time navigation or road-safety system

## Application

The Streamlit application contains:

- Overview
- Temporal Analysis
- Hotspot Explorer
- Intersection Intelligence
- Weather Analysis
- Hourly Incident-Activity Prediction
- Methodology and Limitations

## Author

Final-year undergraduate Computer Science portfolio project.
