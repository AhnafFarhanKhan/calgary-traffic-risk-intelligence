from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import sklearn
import streamlit as st


# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------

st.set_page_config(
    page_title="Calgary Traffic Risk Intelligence",
    page_icon="🚦",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = PROJECT_ROOT / "data" / "cleaned"
MODELS_DIR = PROJECT_ROOT / "models"

FILES = {
    "incidents": CLEAN_DIR / "dashboard_incidents.csv",
    "hotspots": CLEAN_DIR / "hotspot_summary.csv",
    "hotspot_time": CLEAN_DIR / "hotspot_time_profile.csv",
    "controls": CLEAN_DIR / "traffic_control_risk_summary.csv",
    "weather_comparison": CLEAN_DIR / "weather_condition_comparison.csv",
    "temperature_bands": CLEAN_DIR / "temperature_band_summary.csv",
    "metrics": CLEAN_DIR / "model_metrics.csv",
    "importance": CLEAN_DIR / "model_feature_importance.csv",
    "test_predictions": CLEAN_DIR / "model_test_predictions_2025.csv",
    "model": MODELS_DIR / "hourly_incident_model.joblib",
    "metadata": MODELS_DIR / "hourly_incident_model_metadata.json",
}

REQUIRED_FILES = [
    FILES["incidents"],
    FILES["hotspots"],
    FILES["controls"],
    FILES["weather_comparison"],
    FILES["temperature_bands"],
    FILES["metrics"],
    FILES["importance"],
    FILES["model"],
    FILES["metadata"],
]


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------


def stop_if_files_missing(paths):
    missing = [
        str(path.relative_to(PROJECT_ROOT))
        for path in paths
        if not path.exists()
    ]

    if missing:
        st.error("Some required project files are missing.")
        st.code("\n".join(missing))
        st.caption("Run notebooks 01–05 in order and then restart the app.")
        st.stop()


@st.cache_data
def load_csv(path_str, modified_time):
    del modified_time
    return pd.read_csv(path_str, low_memory=False)


@st.cache_data
def load_json(path_str, modified_time):
    del modified_time
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


@st.cache_resource
def load_model(path_str, modified_time):
    del modified_time
    return joblib.load(path_str)


def read_csv(path):
    return load_csv(str(path), path.stat().st_mtime_ns)


def read_json(path):
    return load_json(str(path), path.stat().st_mtime_ns)


def read_model(path):
    return load_model(str(path), path.stat().st_mtime_ns)


def require_columns(df, columns, filename):
    missing = sorted(set(columns) - set(df.columns))

    if missing:
        st.error(f"{filename} is missing required columns.")
        st.code("\n".join(missing))
        st.caption("Rerun the notebooks so the app and generated files match.")
        st.stop()


def restore_local_datetime(series):
    return pd.to_datetime(
        series,
        utc=True,
        errors="coerce",
    ).dt.tz_convert("America/Edmonton")


def clean_chart(fig, height=None):
    fig.update_layout(
        margin=dict(l=10, r=10, t=45, b=10),
        legend_title_text="",
        height=height,
    )
    return fig


def show_chart(fig):
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False},
    )


def format_int(value):
    if pd.isna(value):
        return "—"
    return f"{int(round(float(value))):,}"


def safe_text(value, fallback="—"):
    if pd.isna(value) or str(value).strip() == "":
        return fallback
    return str(value)


def build_prediction_row(
    feature_columns,
    hour,
    weekday_number,
    month,
    temp_c,
    rel_hum_pct,
    wind_speed_kmh,
    visibility_km,
    snow_flag,
    rain_flag,
):
    freezing_temp = int(temp_c <= 0)
    low_visibility = int(visibility_km < 5)

    adverse_weather_flag = int(
        bool(snow_flag)
        or bool(rain_flag)
        or freezing_temp
        or low_visibility
        or wind_speed_kmh >= 50
    )

    row = pd.DataFrame(
        [{
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dow_sin": np.sin(2 * np.pi * weekday_number / 7),
            "dow_cos": np.cos(2 * np.pi * weekday_number / 7),
            "month_sin": np.sin(2 * np.pi * (month - 1) / 12),
            "month_cos": np.cos(2 * np.pi * (month - 1) / 12),
            "weekend_feature": int(weekday_number >= 5),
            "rush_hour": int(6 <= hour <= 9 or 15 <= hour <= 18),
            "temp_c": temp_c,
            "rel_hum_pct": rel_hum_pct,
            "wind_speed_kmh": wind_speed_kmh,
            "visibility_km": visibility_km,
            "freezing_temp": freezing_temp,
            "low_visibility": low_visibility,
            "snow_flag": int(bool(snow_flag)),
            "rain_flag": int(bool(rain_flag)),
            "adverse_weather_flag": adverse_weather_flag,
        }]
    )

    missing = [name for name in feature_columns if name not in row.columns]
    if missing:
        raise ValueError(
            "The saved model expects unsupported features: "
            + ", ".join(missing)
        )

    return row[feature_columns]


# ---------------------------------------------------------
# Load and validate project outputs
# ---------------------------------------------------------

stop_if_files_missing(REQUIRED_FILES)

try:
    incidents = read_csv(FILES["incidents"])
    hotspots = read_csv(FILES["hotspots"])
    controls = read_csv(FILES["controls"])
    weather_comparison = read_csv(FILES["weather_comparison"])
    temperature_bands = read_csv(FILES["temperature_bands"])
    model_metrics = read_csv(FILES["metrics"])
    feature_importance = read_csv(FILES["importance"])
    metadata = read_json(FILES["metadata"])
except Exception as exc:
    st.error("A project data file could not be loaded.")
    st.code(str(exc))
    st.stop()

try:
    model = read_model(FILES["model"])
except Exception as exc:
    st.error("The trained model could not be loaded.")
    st.code(str(exc))
    st.caption("Rerun Notebook 05 and use the matching scikit-learn version.")
    st.stop()

hotspot_time = (
    read_csv(FILES["hotspot_time"])
    if FILES["hotspot_time"].exists()
    else pd.DataFrame()
)

test_predictions = (
    read_csv(FILES["test_predictions"])
    if FILES["test_predictions"].exists()
    else pd.DataFrame()
)

require_columns(
    incidents,
    {
        "incident_id",
        "description",
        "start_dt_local",
        "year",
        "month",
        "month_name",
        "weekday",
        "hour",
        "time_period",
        "hotspot_id",
    },
    "dashboard_incidents.csv",
)

require_columns(
    hotspots,
    {
        "hotspot_id",
        "incident_count",
        "center_latitude",
        "center_longitude",
    },
    "hotspot_summary.csv",
)

require_columns(
    controls,
    {
        "signal_id",
        "intersection_name",
        "control_type",
        "latitude",
        "longitude",
        "historical_incident_risk_index",
    },
    "traffic_control_risk_summary.csv",
)

require_columns(
    weather_comparison,
    {"condition", "relative_rate"},
    "weather_condition_comparison.csv",
)

require_columns(
    temperature_bands,
    {"temperature_band", "mean_incidents_per_hour"},
    "temperature_band_summary.csv",
)

require_columns(
    model_metrics,
    {"model", "MAE", "RMSE", "R2", "Mean Poisson Deviance"},
    "model_metrics.csv",
)

require_columns(
    feature_importance,
    {"feature", "importance_mean"},
    "model_feature_importance.csv",
)

if not isinstance(metadata.get("feature_columns"), list):
    st.error("Model metadata does not contain a valid feature list.")
    st.stop()

incidents = incidents.copy()
incidents["start_dt_local"] = restore_local_datetime(
    incidents["start_dt_local"]
)

trained_version = metadata.get("scikit_learn_version")
if trained_version and trained_version != sklearn.__version__:
    st.warning(
        f"Model trained with scikit-learn {trained_version}; "
        f"current environment uses {sklearn.__version__}."
    )


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("Calgary Traffic Risk Intelligence")
st.caption(
    "Historical traffic patterns, geographic hotspots, weather associations, "
    "intersection-level indicators, and hourly incident-activity modelling."
)

st.divider()


# ---------------------------------------------------------
# Navigation
# ---------------------------------------------------------

overview_tab, hotspot_tab, intersection_tab, weather_tab, prediction_tab = st.tabs(
    ["Overview", "Hotspots", "Intersections", "Weather", "Prediction"]
)


# ---------------------------------------------------------
# Overview
# ---------------------------------------------------------

with overview_tab:
    start_date = incidents["start_dt_local"].min()
    end_date = incidents["start_dt_local"].max()

    cols = st.columns(4)
    cols[0].metric("Recorded incidents", f"{len(incidents):,}")
    cols[1].metric("DBSCAN hotspots", f"{len(hotspots):,}")
    cols[2].metric("Traffic-control locations", f"{len(controls):,}")
    cols[3].metric("Years", f"{incidents['year'].nunique():,}")

    if pd.notna(start_date) and pd.notna(end_date):
        st.caption(
            f"Archive period: {start_date.date()} to {end_date.date()}. "
            "Boundary years and source-collection gaps should be interpreted cautiously."
        )

    available_years = sorted(
        incidents["year"].dropna().astype(int).unique().tolist()
    )

    if available_years:
        year_range = st.slider(
            "Year range",
            min_value=min(available_years),
            max_value=max(available_years),
            value=(min(available_years), max(available_years)),
        )

        filtered = incidents[
            incidents["year"].between(year_range[0], year_range[1])
        ].copy()
    else:
        filtered = incidents.copy()

    yearly = (
        filtered.groupby("year")
        .size()
        .reset_index(name="incident_count")
    )

    fig = px.line(
        yearly,
        x="year",
        y="incident_count",
        markers=True,
        title="Recorded incidents by year",
        labels={"year": "Year", "incident_count": "Recorded incidents"},
    )
    fig.update_xaxes(dtick=1)
    show_chart(clean_chart(fig))

    left, right = st.columns(2)

    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    with left:
        weekday_counts = (
            filtered["weekday"]
            .value_counts()
            .reindex(weekday_order)
            .dropna()
            .rename_axis("weekday")
            .reset_index(name="incident_count")
        )

        fig = px.bar(
            weekday_counts,
            x="weekday",
            y="incident_count",
            title="By weekday",
            labels={"weekday": "Weekday", "incident_count": "Recorded incidents"},
        )
        show_chart(clean_chart(fig, height=360))

    with right:
        hourly = (
            filtered.groupby("hour")
            .size()
            .reset_index(name="incident_count")
        )

        fig = px.line(
            hourly,
            x="hour",
            y="incident_count",
            markers=True,
            title="By hour of day",
            labels={"hour": "Hour", "incident_count": "Recorded incidents"},
        )
        fig.update_xaxes(dtick=2)
        show_chart(clean_chart(fig, height=360))


# ---------------------------------------------------------
# Hotspots
# ---------------------------------------------------------

with hotspot_tab:
    st.subheader("Recurring incident hotspots")
    st.caption("Hotspots are DBSCAN clusters of historically nearby recorded incidents.")

    map_hotspots = hotspots.dropna(
        subset=["center_latitude", "center_longitude", "incident_count"]
    ).copy()

    if map_hotspots.empty:
        st.info("No hotspot coordinates are available.")
    else:
        hover_data = {
            "incident_count": True,
            "center_latitude": False,
            "center_longitude": False,
        }

        for optional_column in [
            "most_common_weekday",
            "most_common_hour",
            "most_common_time_period",
        ]:
            if optional_column in map_hotspots.columns:
                hover_data[optional_column] = True

        fig = px.scatter_map(
            map_hotspots,
            lat="center_latitude",
            lon="center_longitude",
            size="incident_count",
            hover_name="hotspot_id",
            hover_data=hover_data,
            zoom=9,
            map_style="open-street-map",
            size_max=26,
        )
        show_chart(clean_chart(fig, height=570))

    hotspot_ids = sorted(
        hotspots["hotspot_id"].dropna().astype(int).unique().tolist()
    )

    if hotspot_ids:
        selected_hotspot = st.selectbox(
            "Inspect hotspot",
            hotspot_ids,
        )

        selected_rows = hotspots[
            hotspots["hotspot_id"].astype(int) == selected_hotspot
        ]

        if not selected_rows.empty:
            row = selected_rows.iloc[0]
            cols = st.columns(4)
            cols[0].metric("Incidents", format_int(row.get("incident_count")))
            cols[1].metric("Active years", format_int(row.get("active_years")))
            cols[2].metric(
                "Common weekday",
                safe_text(row.get("most_common_weekday")),
            )

            common_hour = row.get("most_common_hour", np.nan)
            cols[3].metric(
                "Common hour",
                f"{int(common_hour):02d}:00" if pd.notna(common_hour) else "—",
            )

        if not hotspot_time.empty and {
            "hotspot_id",
            "hour",
            "incident_count",
        }.issubset(hotspot_time.columns):
            profile = hotspot_time[
                hotspot_time["hotspot_id"].astype(int) == selected_hotspot
            ]

            if not profile.empty:
                hour_profile = (
                    profile.groupby("hour")["incident_count"]
                    .sum()
                    .reset_index()
                )

                fig = px.line(
                    hour_profile,
                    x="hour",
                    y="incident_count",
                    markers=True,
                    title="Historical activity by hour",
                    labels={"hour": "Hour", "incident_count": "Recorded incidents"},
                )
                fig.update_xaxes(dtick=2)
                show_chart(clean_chart(fig, height=330))


# ---------------------------------------------------------
# Intersections
# ---------------------------------------------------------

with intersection_tab:
    st.subheader("Traffic-control locations")
    st.caption(
        "The Historical Incident Risk Index is a project-defined historical indicator, "
        "not an official safety rating or collision probability."
    )

    risk_column = "historical_incident_risk_index"
    risk_controls = controls.dropna(
        subset=[risk_column, "latitude", "longitude"]
    ).copy()

    if risk_controls.empty:
        st.info("No traffic-control risk records are available.")
    else:
        fig = px.scatter_map(
            risk_controls,
            lat="latitude",
            lon="longitude",
            size=risk_column,
            hover_name="intersection_name",
            hover_data={
                risk_column: True,
                "control_type": True,
                "latitude": False,
                "longitude": False,
            },
            zoom=9,
            map_style="open-street-map",
            size_max=22,
        )
        show_chart(clean_chart(fig, height=540))

        risk_controls["selector_label"] = (
            risk_controls["intersection_name"]
            .fillna("Unnamed location")
            .astype(str)
            + " — "
            + risk_controls["control_type"]
            .fillna("Unknown control")
            .astype(str)
            + " — ID "
            + risk_controls["signal_id"].apply(format_int)
        )

        selected_label = st.selectbox(
            "Inspect location",
            sorted(risk_controls["selector_label"].tolist()),
        )

        selected = risk_controls[
            risk_controls["selector_label"] == selected_label
        ].iloc[0]

        cols = st.columns(4)
        cols[0].metric("Risk index", f"{selected[risk_column]:.1f} / 100")
        cols[1].metric(
            "Historical incidents",
            format_int(selected.get("historical_incident_count")),
        )
        cols[2].metric(
            "Recent incidents",
            format_int(selected.get("recent_incident_count")),
        )
        cols[3].metric("Control type", safe_text(selected.get("control_type")))

        factors = safe_text(
            selected.get("top_risk_factors"),
            "No factor summary available.",
        )
        st.write(f"**Main contributors:** {factors}")

        component_map = {
            "Historical": "historical_score",
            "Recent": "recent_score",
            "Hotspot": "hotspot_score",
            "Weather": "adverse_weather_score",
            "Rush hour": "rush_hour_score",
            "Trend": "trend_score",
        }

        component_rows = [
            {"component": label, "score": selected[column]}
            for label, column in component_map.items()
            if column in selected.index and pd.notna(selected[column])
        ]

        if component_rows:
            component_df = pd.DataFrame(component_rows)
            fig = px.bar(
                component_df,
                x="component",
                y="score",
                title="Risk-index components",
                labels={"component": "Component", "score": "Score"},
            )
            show_chart(clean_chart(fig, height=330))


# ---------------------------------------------------------
# Weather
# ---------------------------------------------------------

with weather_tab:
    st.subheader("Weather associations")
    st.caption(
        "Rates use observed hours as exposure and exclude inferred archive-gap hours. "
        "These are associations, not causal effects."
    )

    comparison_plot = (
        weather_comparison.dropna(subset=["condition", "relative_rate"])
        .sort_values("relative_rate")
        .copy()
    )

    if comparison_plot.empty:
        st.info("No weather-comparison data is available.")
    else:
        fig = px.bar(
            comparison_plot,
            x="relative_rate",
            y="condition",
            orientation="h",
            title="Relative mean incidents per observed hour",
            labels={"relative_rate": "Relative rate", "condition": "Condition"},
        )
        fig.add_vline(x=1, line_dash="dash")
        show_chart(clean_chart(fig, height=430))
        st.caption("1.0 means the condition and comparison hours had the same historical mean rate.")

    temp_plot = temperature_bands.dropna(
        subset=["temperature_band", "mean_incidents_per_hour"]
    ).copy()

    if not temp_plot.empty:
        fig = px.bar(
            temp_plot,
            x="temperature_band",
            y="mean_incidents_per_hour",
            title="Mean recorded incidents per hour by temperature",
            labels={
                "temperature_band": "Temperature",
                "mean_incidents_per_hour": "Mean incidents per hour",
            },
        )
        show_chart(clean_chart(fig, height=370))


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

with prediction_tab:
    st.subheader("Hourly incident-activity estimate")
    st.caption(
        "The model estimates the expected citywide number of recorded incidents during an hour. "
        "It does not predict individual crashes."
    )

    baseline_mae = metadata.get("baseline_mae")
    model_mae = metadata.get("model_mae")
    improvement = metadata.get("mae_improvement_percent")

    metric_cols = st.columns(3)
    metric_cols[0].metric(
        "Baseline MAE",
        f"{baseline_mae:.3f}" if isinstance(baseline_mae, (int, float)) else "—",
    )
    metric_cols[1].metric(
        "Model MAE",
        f"{model_mae:.3f}" if isinstance(model_mae, (int, float)) else "—",
    )
    metric_cols[2].metric(
        "MAE improvement",
        f"{improvement:.1f}%" if isinstance(improvement, (int, float)) else "—",
    )

    weekday_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    month_names = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    with st.form("prediction_form"):
        left, middle, right = st.columns(3)

        with left:
            weekday_name = st.selectbox("Weekday", weekday_names)
            hour = st.slider("Hour", 0, 23, 16)
            month_name = st.selectbox("Month", month_names, index=5)

        with middle:
            temp_c = st.number_input("Temperature (°C)", value=10.0, step=1.0)
            rel_hum_pct = st.number_input(
                "Relative humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=60.0,
                step=1.0,
            )
            wind_speed_kmh = st.number_input(
                "Wind speed (km/h)",
                min_value=0.0,
                value=15.0,
                step=1.0,
            )

        with right:
            visibility_km = st.number_input(
                "Visibility (km)",
                min_value=0.0,
                value=24.0,
                step=1.0,
            )
            snow_flag = st.checkbox("Snow / freezing precipitation")
            rain_flag = st.checkbox("Rain / drizzle / thunderstorm")

        submitted = st.form_submit_button("Estimate")

    if submitted:
        try:
            prediction_row = build_prediction_row(
                feature_columns=metadata["feature_columns"],
                hour=hour,
                weekday_number=weekday_names.index(weekday_name),
                month=month_names.index(month_name) + 1,
                temp_c=temp_c,
                rel_hum_pct=rel_hum_pct,
                wind_speed_kmh=wind_speed_kmh,
                visibility_km=visibility_km,
                snow_flag=snow_flag,
                rain_flag=rain_flag,
            )

            estimate = float(model.predict(prediction_row)[0])
            estimate = max(0.0, estimate)

            st.metric(
                "Estimated recorded incidents during the hour",
                f"{estimate:.2f}",
            )
        except Exception as exc:
            st.error("The prediction could not be generated.")
            st.code(str(exc))

    with st.expander("Model details"):
        st.dataframe(
            model_metrics.round(4),
            use_container_width=True,
            hide_index=True,
        )

        top_importance = (
            feature_importance.dropna(subset=["importance_mean"])
            .head(10)
            .sort_values("importance_mean")
        )

        if not top_importance.empty:
            fig = px.bar(
                top_importance,
                x="importance_mean",
                y="feature",
                orientation="h",
                title="Top permutation feature importance",
                labels={"importance_mean": "Importance", "feature": "Feature"},
            )
            show_chart(clean_chart(fig, height=360))

        if not test_predictions.empty and {
            "month",
            "incident_count",
            "model_prediction",
            "baseline_prediction",
        }.issubset(test_predictions.columns):
            monthly = (
                test_predictions.groupby("month")
                .agg(
                    Actual=("incident_count", "sum"),
                    Model=("model_prediction", "sum"),
                    Baseline=("baseline_prediction", "sum"),
                )
                .reset_index()
                .melt(
                    id_vars="month",
                    var_name="Series",
                    value_name="Incidents",
                )
            )

            fig = px.line(
                monthly,
                x="month",
                y="Incidents",
                color="Series",
                markers=True,
                title="2025 monthly held-out evaluation",
                labels={"month": "Month"},
            )
            fig.update_xaxes(dtick=1)
            show_chart(clean_chart(fig, height=350))


# ---------------------------------------------------------
# Small footer instead of a separate methodology page
# ---------------------------------------------------------

st.divider()

with st.expander("Methodology and limitations"):
    st.markdown(
        """
**Pipeline:** City of Calgary traffic incidents and traffic controls → cleaning and local-time features → Haversine DBSCAN hotspots → nearest-control matching → ECCC hourly weather → historical risk index → chronological machine-learning evaluation → Streamlit dashboard.

**Important limitations:**
- Recorded incidents are not collision probabilities.
- Traffic-volume exposure is not included.
- Calgary airport weather is used as a citywide weather proxy.
- Weather relationships are observational, not causal.
- DBSCAN and nearest-control results depend on project-selected parameters.
- The Historical Incident Risk Index is project-defined and not an official City safety rating.
- The model estimates citywide hourly recorded incident activity, not individual crashes or real-time road safety.
        """
    )

    
