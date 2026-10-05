# Calgary Traffic Risk Intelligence

A data science and machine learning project that analyzes Calgary traffic incidents using temporal patterns, geographic hotspots, traffic-control locations, historical weather, and predictive modelling.

## What This Project Does

The project uses public City of Calgary traffic data and Environment and Climate Change Canada weather data to explore:

- when traffic incidents are recorded most often
- where recurring incident hotspots appear
- which traffic-control locations have higher historical incident concentration
- how incident frequency differs under weather conditions such as snow, freezing temperatures, and low visibility
- whether time and weather features can help estimate hourly citywide incident activity

The project analyzes more than 64,000 historical traffic incident records.

## Project Workflow

```text
Raw traffic + traffic-control data
            ↓
      Data cleaning
            ↓
   Exploratory analysis
            ↓
 DBSCAN hotspot detection
            ↓
Nearest traffic-control matching
            ↓
 Historical weather integration
            ↓
 Weather + risk analysis
            ↓
 Machine-learning model
            ↓
   Streamlit dashboard
```

## Main Methods

### Data Preparation
Traffic incident timestamps are converted to Calgary local time and features such as year, month, weekday, hour, season, and time period are created.

### Hotspot Detection
DBSCAN with Haversine distance is used to find recurring geographic concentrations of incidents without requiring the number of clusters in advance.

### Intersection Matching
Each incident is matched to its nearest traffic-control location using BallTree and Haversine distance. A distance threshold is used so distant incidents are not incorrectly treated as intersection incidents.

### Weather Analysis
Historical hourly weather from Environment and Climate Change Canada is matched to traffic incidents. Comparisons use observed hours so weather conditions are compared more fairly.

### Historical Incident Risk Index
A transparent 0–100 project-defined index summarizes factors such as:

- long-term incident activity
- recent incident activity
- hotspot concentration
- adverse-weather association
- rush-hour concentration
- historical trend

This is **not an official City of Calgary safety score** and does not represent collision probability.

### Predictive Modelling
A `HistGradientBoostingRegressor` with Poisson loss estimates the expected number of recorded traffic incidents across Calgary during an hour.

The model is trained on earlier years and evaluated on a later year using a chronological split. It is also compared with a historical weekday/hour baseline.

## Streamlit Dashboard

The interactive app includes:

- Overview
- Temporal Analysis
- Hotspot Explorer
- Intersection Intelligence
- Weather Analysis
- Hourly Incident-Activity Prediction
- Methodology and Limitations

Run it locally with:

```bash
python -m streamlit run app/app.py
```

## Project Structure

```text
calgary-traffic-risk-intelligence/
│
├── app/
│   └── app.py
│
├── data/
│   └── cleaned/
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
├── requirements.txt
└── README.md
```

## Technologies

Python, Pandas, NumPy, Scikit-learn, Plotly, Streamlit, DBSCAN, BallTree, Jupyter Notebook, Joblib

## Data Sources

- City of Calgary Open Data — Traffic Incidents
- City of Calgary Open Data — Traffic Signals
- Environment and Climate Change Canada — Historical Climate Data

## Important Limitations

- Recorded incidents are not the same as collision probability.
- Traffic-volume exposure is not included.
- Calgary airport weather is used as a citywide weather proxy.
- Weather relationships are observational and do not prove causation.
- DBSCAN and intersection matching depend on project-selected parameters.
- The prediction model estimates citywide hourly incident activity, not individual crashes.
- The application is not a real-time navigation or road-safety system.

## Author

Undergraduate Computer Science portfolio project.
