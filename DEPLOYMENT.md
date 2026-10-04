# Deployment Guide

## 1. Test Locally First

From the project root:

```bash
streamlit run app/app.py
```

Open the local URL printed by Streamlit.

Test every tab:

- Overview
- Temporal Analysis
- Hotspot Explorer
- Intersection Intelligence
- Weather Analysis
- Prediction
- Methodology

## 2. Confirm Required Files

The deployed repository must contain the cleaned application datasets and
trained model files used by `app/app.py`.

At minimum:

```text
data/cleaned/incidents_weather.csv
data/cleaned/hotspot_summary.csv
data/cleaned/hotspot_time_profile.csv
data/cleaned/traffic_control_risk_summary.csv
data/cleaned/hourly_incident_weather.csv
data/cleaned/weather_condition_comparison.csv
data/cleaned/model_metrics.csv
data/cleaned/model_feature_importance.csv
data/cleaned/model_test_predictions_2025.csv

models/hourly_incident_model.joblib
models/hourly_incident_model_metadata.json
```

If any individual file exceeds GitHub's normal file-size limit, either:

1. reduce it to only the columns needed by the application, or
2. use Git LFS.

Do not commit `.venv/`.

## 3. Check the Model Environment

A saved scikit-learn model should ideally be loaded using the same
scikit-learn version used to train it.

Before deployment, check:

```bash
python -c "import sklearn; print(sklearn.__version__)"
```

If deployment reports a model-version error, pin that exact version in
`requirements.txt`.

## 4. Push to GitHub

Your repository should contain:

```text
app/
data/cleaned/
models/
notebooks/
README.md
requirements.txt
```

Raw government source files do not need to be included if you prefer to keep
the repository smaller.

## 5. Deploy on Streamlit Community Cloud

Create a new Streamlit application from the GitHub repository.

Use:

```text
Main file path: app/app.py
```

Streamlit installs dependencies from the root `requirements.txt`.

## 6. Final Deployment Test

Check:

- all CSV files load
- the model loads
- maps display
- charts display
- hotspot selection works
- intersection selection works
- prediction form works
- no local absolute file paths remain

## 7. Screenshots

Capture at least:

1. Overview page
2. Hotspot map
3. Intersection Intelligence page
4. Weather Analysis page
5. Prediction page

Add the strongest 2–3 screenshots to the GitHub README.
