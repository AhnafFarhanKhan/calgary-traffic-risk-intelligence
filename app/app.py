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


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def require_files(paths):
    """Stop with a clear message if required project outputs are missing."""

    missing = [
        str(path.relative_to(PROJECT_ROOT))
        for path in paths
        if not path.exists()
    ]

    if missing:
        st.error(
            "Required project files are missing. "
            "Run notebooks 01–05 in order first."
        )

        st.code(
            "\n".join(missing)
        )

        st.stop()


@st.cache_data
def load_csv(path, file_version):
    """Load a CSV and refresh the cache when the file changes."""

    return pd.read_csv(
        path,
        low_memory=False,
    )


@st.cache_resource
def load_model(path, file_version):
    """Load the trained model and refresh when the artifact changes."""

    return joblib.load(path)


@st.cache_data
def load_metadata(path, file_version):
    """Load model metadata and refresh when the file changes."""

    return json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )


def version_of(path):
    return path.stat().st_mtime_ns


def require_columns(df, required, label):
    missing = sorted(
        set(required) - set(df.columns)
    )

    if missing:
        st.error(
            f"{label} is missing required columns. "
            "Rerun notebooks 01–05 in order."
        )
        st.code("\n".join(missing))
        st.stop()


def restore_incident_datetimes(df):
    """Restore datetimes saved inside CSV files."""

    result = df.copy()

    if "start_dt_utc" in result.columns:
        result["start_dt_utc"] = pd.to_datetime(
            result["start_dt_utc"],
            utc=True,
            errors="coerce",
        )

    if "start_dt_local" in result.columns:
        result["start_dt_local"] = pd.to_datetime(
            result["start_dt_local"],
            utc=True,
            errors="coerce",
        ).dt.tz_convert(
            "America/Edmonton"
        )

    return result


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
    """Create one model-ready hourly scenario."""

    freezing_temp = int(
        temp_c <= 0
    )

    low_visibility = int(
        visibility_km < 5
    )

    adverse_weather_flag = int(
        bool(snow_flag)
        or bool(rain_flag)
        or freezing_temp
        or low_visibility
        or wind_speed_kmh >= 50
    )

    row = pd.DataFrame(
        [
            {
                "hour_sin":
                    np.sin(
                        2 * np.pi * hour / 24
                    ),
                "hour_cos":
                    np.cos(
                        2 * np.pi * hour / 24
                    ),
                "dow_sin":
                    np.sin(
                        2 * np.pi
                        * weekday_number / 7
                    ),
                "dow_cos":
                    np.cos(
                        2 * np.pi
                        * weekday_number / 7
                    ),
                "month_sin":
                    np.sin(
                        2 * np.pi
                        * (month - 1) / 12
                    ),
                "month_cos":
                    np.cos(
                        2 * np.pi
                        * (month - 1) / 12
                    ),
                "weekend_feature":
                    int(
                        weekday_number >= 5
                    ),
                "rush_hour":
                    int(
                        6 <= hour <= 9
                        or 15 <= hour <= 18
                    ),
                "temp_c": temp_c,
                "rel_hum_pct": rel_hum_pct,
                "wind_speed_kmh":
                    wind_speed_kmh,
                "visibility_km":
                    visibility_km,
                "freezing_temp":
                    freezing_temp,
                "low_visibility":
                    low_visibility,
                "snow_flag":
                    int(bool(snow_flag)),
                "rain_flag":
                    int(bool(rain_flag)),
                "adverse_weather_flag":
                    adverse_weather_flag,
            }
        ]
    )

    missing_features = [
        column
        for column in feature_columns
        if column not in row.columns
    ]

    if missing_features:
        raise ValueError(
            "Model metadata requests unsupported features: "
            f"{missing_features}. Rerun Notebook 05 and redeploy "
            "the matching model/metadata files."
        )

    return row[
        feature_columns
    ]

def format_int(value):
    if pd.isna(value):
        return "—"

    return f"{int(value):,}"


# ---------------------------------------------------------
# Load project outputs
# ---------------------------------------------------------

core_paths = [
    FILES["incidents"],
    FILES["hotspots"],
    FILES["hotspot_time"],
    FILES["controls"],
    FILES["weather_comparison"],
    FILES["temperature_bands"],
    FILES["metrics"],
    FILES["importance"],
    FILES["test_predictions"],
    FILES["model"],
    FILES["metadata"],
]

require_files(core_paths)

incidents = restore_incident_datetimes(
    load_csv(
        FILES["incidents"],
        version_of(FILES["incidents"]),
    )
)

hotspots = load_csv(
    FILES["hotspots"],
    version_of(FILES["hotspots"]),
)

controls = load_csv(
    FILES["controls"],
    version_of(FILES["controls"]),
)

weather_comparison = load_csv(
    FILES["weather_comparison"],
    version_of(FILES["weather_comparison"]),
)

temperature_bands = load_csv(
    FILES["temperature_bands"],
    version_of(FILES["temperature_bands"]),
)

model_metrics = load_csv(
    FILES["metrics"],
    version_of(FILES["metrics"]),
)

feature_importance = load_csv(
    FILES["importance"],
    version_of(FILES["importance"]),
)

metadata = load_metadata(
    FILES["metadata"],
    version_of(FILES["metadata"]),
)

try:
    model = load_model(
        FILES["model"],
        version_of(FILES["model"]),
    )
except Exception as exc:
    st.error(
        "The trained model could not be loaded. "
        "Rerun Notebook 05 using the scikit-learn version in requirements.txt."
    )
    st.code(str(exc))
    st.stop()

trained_sklearn_version = metadata.get(
    "scikit_learn_version"
)

if (
    trained_sklearn_version
    and trained_sklearn_version
    != sklearn.__version__
):
    st.warning(
        "The deployed scikit-learn version "
        f"({sklearn.__version__}) differs from the model's "
        f"training version ({trained_sklearn_version}). "
        "Pin the training version in requirements.txt."
    )

hotspot_time = load_csv(
    FILES["hotspot_time"],
    version_of(FILES["hotspot_time"]),
)

test_predictions = load_csv(
    FILES["test_predictions"],
    version_of(FILES["test_predictions"]),
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
    {
        "temperature_band",
        "observed_hours",
        "mean_incidents_per_hour",
    },
    "temperature_band_summary.csv",
)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title(
    "🚦 Calgary Traffic Risk Intelligence"
)

st.caption(
    "Historical traffic incidents, geographic hotspots, "
    "weather associations, intersection-level risk features, "
    "and hourly incident-activity modelling."
)

st.info(
    "This application is an analytical project built from "
    "historical open data. The Historical Incident Risk Index "
    "is a project-defined metric, not an official City of Calgary "
    "safety rating."
)


# ---------------------------------------------------------
# Main navigation
# ---------------------------------------------------------

tabs = st.tabs(
    [
        "Overview",
        "Temporal Analysis",
        "Hotspot Explorer",
        "Intersection Intelligence",
        "Weather Analysis",
        "Prediction",
        "Methodology",
    ]
)


# =========================================================
# Overview
# =========================================================

with tabs[0]:
    st.header(
        "Project Overview"
    )

    total_incidents = len(
        incidents
    )

    hotspot_count = (
        incidents.loc[
            incidents["hotspot_id"] != -1,
            "hotspot_id",
        ]
        .nunique()
        if "hotspot_id" in incidents.columns
        else len(hotspots)
    )

    start_date = (
        incidents["start_dt_local"]
        .min()
    )

    end_date = (
        incidents["start_dt_local"]
        .max()
    )

    metric_cols = st.columns(4)

    metric_cols[0].metric(
        "Recorded incidents",
        f"{total_incidents:,}",
    )

    metric_cols[1].metric(
        "DBSCAN hotspots",
        f"{hotspot_count:,}",
    )

    metric_cols[2].metric(
        "Traffic-control locations",
        f"{len(controls):,}",
    )

    metric_cols[3].metric(
        "Years represented",
        f"{incidents['year'].nunique():,}",
    )

    st.write(
        "**Incident period:**",
        start_date.date()
        if pd.notna(start_date)
        else "Unknown",
        "→",
        end_date.date()
        if pd.notna(end_date)
        else "Unknown",
    )

    st.caption(
        "Annual values are recorded archive counts. "
        "The City source can contain collection gaps, and "
        "2016/2026 are partial boundary years."
    )

    st.subheader(
        "Annual recorded incidents"
    )

    yearly = (
        incidents
        .groupby("year")
        .size()
        .reset_index(
            name="incident_count"
        )
    )

    fig = px.line(
        yearly,
        x="year",
        y="incident_count",
        markers=True,
        labels={
            "year": "Year",
            "incident_count":
                "Recorded incidents",
        },
    )

    fig.update_xaxes(
        dtick=1
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(
            "Incidents by time period"
        )

        order = [
            "Night",
            "Morning Rush",
            "Midday",
            "Evening Rush",
            "Evening",
        ]

        time_counts = (
            incidents["time_period"]
            .value_counts()
            .reindex(order)
            .dropna()
            .rename_axis(
                "time_period"
            )
            .reset_index(
                name="incident_count"
            )
        )

        fig = px.bar(
            time_counts,
            x="time_period",
            y="incident_count",
            labels={
                "time_period":
                    "Time period",
                "incident_count":
                    "Recorded incidents",
            },
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with col2:
        st.subheader(
            "Top source descriptions"
        )

        top_descriptions = (
            incidents["description"]
            .value_counts()
            .head(10)
            .rename_axis(
                "description"
            )
            .reset_index(
                name="incident_count"
            )
            .sort_values(
                "incident_count"
            )
        )

        fig = px.bar(
            top_descriptions,
            x="incident_count",
            y="description",
            orientation="h",
            labels={
                "description":
                    "Description",
                "incident_count":
                    "Recorded incidents",
            },
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# =========================================================
# Temporal Analysis
# =========================================================

with tabs[1]:
    st.header(
        "Temporal Analysis"
    )

    available_years = sorted(
        incidents["year"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    selected_years = st.multiselect(
        "Years",
        options=available_years,
        default=available_years,
    )

    filtered = incidents[
        incidents["year"].isin(
            selected_years
        )
    ].copy()

    if filtered.empty:
        st.warning(
            "No incidents match the selected years."
        )
    else:
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
                .reindex(
                    weekday_order
                )
                .dropna()
                .rename_axis(
                    "weekday"
                )
                .reset_index(
                    name="incident_count"
                )
            )

            fig = px.bar(
                weekday_counts,
                x="weekday",
                y="incident_count",
                title="Recorded incidents by weekday",
                labels={
                    "weekday":
                        "Weekday",
                    "incident_count":
                        "Recorded incidents",
                },
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        with right:
            monthly = (
                filtered
                .groupby(
                    [
                        "month",
                        "month_name",
                    ]
                )
                .size()
                .reset_index(
                    name="incident_count"
                )
                .sort_values(
                    "month"
                )
            )

            fig = px.bar(
                monthly,
                x="month_name",
                y="incident_count",
                title="Recorded incidents by month",
                labels={
                    "month_name":
                        "Month",
                    "incident_count":
                        "Recorded incidents",
                },
            )

            fig.update_xaxes(
                categoryorder="array",
                categoryarray=monthly[
                    "month_name"
                ],
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        hourly = (
            filtered
            .groupby("hour")
            .size()
            .reset_index(
                name="incident_count"
            )
        )

        fig = px.line(
            hourly,
            x="hour",
            y="incident_count",
            markers=True,
            title="Recorded incidents by hour of day",
            labels={
                "hour": "Hour",
                "incident_count":
                    "Recorded incidents",
            },
        )

        fig.update_xaxes(
            dtick=1
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# =========================================================
# Hotspot Explorer
# =========================================================

with tabs[2]:
    st.header(
        "Hotspot Explorer"
    )

    if hotspots.empty:
        st.warning(
            "No hotspot summary is available."
        )

    else:
        st.write(
            "DBSCAN hotspots represent recurring geographic "
            "concentrations of recorded incidents under the "
            "project's selected clustering parameters."
        )

        map_hotspots = hotspots.copy()

        fig = px.scatter_map(
            map_hotspots,
            lat="center_latitude",
            lon="center_longitude",
            size="incident_count",
            hover_name="hotspot_id",
            hover_data={
                "incident_count": True,
                "most_common_weekday": True,
                "most_common_hour": True,
                "most_common_time_period": True,
                "active_years": True,
                "center_latitude": False,
                "center_longitude": False,
            },
            zoom=9,
            map_style="open-street-map",
            size_max=28,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        hotspot_ids = (
            hotspots["hotspot_id"]
            .astype(int)
            .tolist()
        )

        selected_hotspot = st.selectbox(
            "Inspect hotspot",
            options=hotspot_ids,
        )

        row = (
            hotspots[
                hotspots["hotspot_id"]
                == selected_hotspot
            ]
            .iloc[0]
        )

        cols = st.columns(4)

        cols[0].metric(
            "Historical incidents",
            format_int(
                row.get(
                    "incident_count"
                )
            ),
        )

        cols[1].metric(
            "Active years",
            format_int(
                row.get(
                    "active_years"
                )
            ),
        )

        cols[2].metric(
            "Common weekday",
            str(
                row.get(
                    "most_common_weekday",
                    "—",
                )
            ),
        )

        common_hour = row.get(
            "most_common_hour",
            np.nan,
        )

        cols[3].metric(
            "Common hour",
            (
                f"{int(common_hour):02d}:00"
                if pd.notna(common_hour)
                else "—"
            ),
        )

        if not hotspot_time.empty:
            selected_profile = hotspot_time[
                hotspot_time["hotspot_id"]
                == selected_hotspot
            ].copy()

            if not selected_profile.empty:
                hour_profile = (
                    selected_profile
                    .groupby("hour")[
                        "incident_count"
                    ]
                    .sum()
                    .reset_index()
                )

                fig = px.line(
                    hour_profile,
                    x="hour",
                    y="incident_count",
                    markers=True,
                    title=(
                        f"Hotspot {selected_hotspot}: "
                        "historical activity by hour"
                    ),
                    labels={
                        "hour": "Hour",
                        "incident_count":
                            "Recorded incidents",
                    },
                )

                fig.update_xaxes(
                    dtick=1
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )


# =========================================================
# Intersection Intelligence
# =========================================================

with tabs[3]:
    st.header(
        "Intersection Intelligence"
    )

    st.warning(
        "The Historical Incident Risk Index is a project-defined "
        "historical indicator. It is not an official road-safety score."
    )

    risk_column = (
        "historical_incident_risk_index"
    )

    risk_controls = controls[
        controls[risk_column].notna()
    ].copy()

    if risk_controls.empty:
        st.warning(
            "No traffic-control risk summary is available."
        )

    else:
        fig = px.scatter_map(
            risk_controls,
            lat="latitude",
            lon="longitude",
            size=risk_column,
            hover_name="intersection_name",
            hover_data={
                risk_column: True,
                "historical_incident_count": True,
                "recent_incident_count": True,
                "control_type": True,
                "latitude": False,
                "longitude": False,
            },
            zoom=9,
            map_style="open-street-map",
            size_max=24,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        risk_controls = risk_controls.copy()

        risk_controls["selector_label"] = (
            risk_controls[
                "intersection_name"
            ]
            .fillna("Unnamed location")
            .astype(str)
            + " — "
            + risk_controls[
                "control_type"
            ]
            .fillna("Unknown control")
            .astype(str)
            + " — ID "
            + risk_controls[
                "signal_id"
            ]
            .astype(str)
        )

        selector_labels = (
            risk_controls[
                "selector_label"
            ]
            .sort_values()
            .tolist()
        )

        selected_label = st.selectbox(
            "Inspect traffic-control location",
            options=selector_labels,
        )

        selected = (
            risk_controls[
                risk_controls[
                    "selector_label"
                ] == selected_label
            ]
            .iloc[0]
        )

        metric_cols = st.columns(4)

        metric_cols[0].metric(
            "Historical Incident Risk Index",
            f"{selected[risk_column]:.1f} / 100",
        )

        metric_cols[1].metric(
            "Historical matched incidents",
            format_int(
                selected.get(
                    "historical_incident_count"
                )
            ),
        )

        metric_cols[2].metric(
            "Recent incidents (2024–2025)",
            format_int(
                selected.get(
                    "recent_incident_count"
                )
            ),
        )

        metric_cols[3].metric(
            "Control type",
            str(
                selected.get(
                    "control_type",
                    "—",
                )
            ),
        )

        st.write(
            "**Main score contributors:**",
            selected.get(
                "top_risk_factors",
                "No explanation available.",
            ),
        )

        component_map = {
            "Historical activity":
                "historical_score",
            "Recent activity":
                "recent_score",
            "Hotspot concentration":
                "hotspot_score",
            "Adverse-weather incident share":
                "adverse_weather_score",
            "Rush-hour concentration":
                "rush_hour_score",
            "Coverage-adjusted trend":
                "trend_score",
        }

        component_rows = []

        for label, column in component_map.items():
            if column in selected.index:
                component_rows.append(
                    {
                        "component": label,
                        "score": selected[column],
                    }
                )

        if component_rows:
            component_df = pd.DataFrame(
                component_rows
            )

            fig = px.bar(
                component_df,
                x="component",
                y="score",
                title="Risk-index component scores",
                labels={
                    "component": "Component",
                    "score": "0–100 component score",
                },
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        st.subheader(
            "Highest index values"
        )

        top_controls = (
            risk_controls
            .sort_values(
                risk_column,
                ascending=False,
            )
            .head(20)
            .sort_values(
                risk_column
            )
        )

        fig = px.bar(
            top_controls,
            x=risk_column,
            y="intersection_name",
            orientation="h",
            labels={
                risk_column:
                    "Historical Incident Risk Index",
                "intersection_name":
                    "Traffic-control location",
            },
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# =========================================================
# Weather Analysis
# =========================================================

with tabs[4]:
    st.header(
        "Weather Analysis"
    )

    st.write(
        "Weather comparisons use observed hours as exposure. "
        "Hours inside inferred source-collection gaps are excluded "
        "so obvious archive outages are not treated as true "
        "zero-incident periods."
    )

    comparison_plot = (
        weather_comparison
        .dropna(
            subset=[
                "relative_rate"
            ]
        )
        .copy()
    )

    fig = px.bar(
        comparison_plot,
        x="condition",
        y="relative_rate",
        labels={
            "condition":
                "Weather condition",
            "relative_rate":
                "Relative mean incidents per hour",
        },
        title=(
            "Relative Recorded Incident Frequency "
            "by Weather Condition"
        ),
    )

    fig.add_hline(
        y=1,
        line_dash="dash",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.caption(
        "A value above 1 means the historical mean incidents "
        "per observed hour was higher during that condition. "
        "This does not establish causation."
    )

    temp_rates = temperature_bands.copy()

    fig = px.bar(
        temp_rates,
        x="temperature_band",
        y="mean_incidents_per_hour",
        title=(
            "Mean Recorded Incidents per Hour "
            "by Temperature Band"
        ),
        labels={
            "temperature_band":
                "Temperature",
            "mean_incidents_per_hour":
                "Mean recorded incidents per hour",
        },
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# =========================================================
# Prediction
# =========================================================

with tabs[5]:
    st.header(
        "Hourly Incident-Activity Prediction"
    )

    st.write(
        "The model estimates the expected number of recorded "
        "traffic incidents across Calgary during one hour."
    )

    st.caption(
        "This is not an individual crash prediction and should "
        "not be used as a real-time navigation or safety system. "
        "Model evaluation excludes inferred archive-gap hours."
    )

    metric_view = model_metrics.copy()

    st.subheader(
        "Model evaluation"
    )

    st.dataframe(
        metric_view,
        use_container_width=True,
        hide_index=True,
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

    with st.form(
        "prediction_form"
    ):
        form_cols = st.columns(3)

        with form_cols[0]:
            weekday_name = st.selectbox(
                "Weekday",
                weekday_names,
                index=0,
            )

            hour = st.slider(
                "Hour",
                min_value=0,
                max_value=23,
                value=16,
            )

            month_name = st.selectbox(
                "Month",
                month_names,
                index=5,
            )

        with form_cols[1]:
            temp_c = st.number_input(
                "Temperature (°C)",
                value=10.0,
                step=1.0,
            )

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

        with form_cols[2]:
            visibility_km = st.number_input(
                "Visibility (km)",
                min_value=0.0,
                value=24.0,
                step=1.0,
            )

            snow_flag = st.checkbox(
                "Snow / freezing precipitation in weather text"
            )

            rain_flag = st.checkbox(
                "Rain / drizzle / thunderstorm condition"
            )

        submitted = st.form_submit_button(
            "Estimate incident activity"
        )

    if submitted:
        weekday_number = (
            weekday_names.index(
                weekday_name
            )
        )

        month_number = (
            month_names.index(
                month_name
            )
            + 1
        )

        feature_columns = metadata[
            "feature_columns"
        ]

        prediction_row = (
            build_prediction_row(
                feature_columns=
                    feature_columns,
                hour=hour,
                weekday_number=
                    weekday_number,
                month=month_number,
                temp_c=temp_c,
                rel_hum_pct=
                    rel_hum_pct,
                wind_speed_kmh=
                    wind_speed_kmh,
                visibility_km=
                    visibility_km,
                snow_flag=
                    snow_flag,
                rain_flag=
                    rain_flag,
            )
        )

        estimate = float(
            model.predict(
                prediction_row
            )[0]
        )

        estimate = max(
            0.0,
            estimate,
        )

        st.metric(
            "Estimated recorded incidents during the hour",
            f"{estimate:.2f}",
        )

        st.caption(
            "This is the model's expected citywide count for "
            "the selected historical-pattern scenario."
        )

    st.subheader(
        "Feature importance"
    )

    top_importance = (
        feature_importance
        .head(15)
        .sort_values(
            "importance_mean"
        )
    )

    fig = px.bar(
        top_importance,
        x="importance_mean",
        y="feature",
        orientation="h",
        error_x="importance_std"
        if "importance_std"
        in top_importance.columns
        else None,
        labels={
            "importance_mean":
                "Permutation importance",
            "feature":
                "Feature",
        },
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    if not test_predictions.empty:
        st.subheader(
            "2025 monthly held-out performance"
        )

        st.caption(
            "Months fully inside an inferred collection gap "
            "are absent rather than plotted as zero incidents."
        )

        monthly_test = (
            test_predictions
            .groupby("month")
            .agg(
                actual_incidents=(
                    "incident_count",
                    "sum",
                ),
                predicted_incidents=(
                    "model_prediction",
                    "sum",
                ),
                baseline_incidents=(
                    "baseline_prediction",
                    "sum",
                ),
            )
            .reset_index()
        )

        monthly_long = (
            monthly_test
            .melt(
                id_vars="month",
                value_vars=[
                    "actual_incidents",
                    "predicted_incidents",
                    "baseline_incidents",
                ],
                var_name="series",
                value_name="incident_count",
            )
        )

        fig = px.line(
            monthly_long,
            x="month",
            y="incident_count",
            color="series",
            markers=True,
            labels={
                "month": "Month",
                "incident_count":
                    "Recorded / estimated incidents",
                "series": "Series",
            },
        )

        fig.update_xaxes(
            dtick=1
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# =========================================================
# Methodology
# =========================================================

with tabs[6]:
    st.header(
        "Methodology and Limitations"
    )

    st.markdown(
        """
### Data pipeline

1. City of Calgary Traffic Incidents data
2. City of Calgary Traffic Signals data
3. Cleaning and Calgary-local-time feature engineering
4. Exploratory temporal analysis
5. Haversine DBSCAN hotspot detection
6. BallTree nearest traffic-control matching
7. Environment and Climate Change Canada hourly weather
8. Weather-exposure analysis with inferred archive-gap filtering and compact dashboard summaries
9. Historical Incident Risk Index
10. Chronological machine-learning evaluation on eligible observed hours
11. Interactive Streamlit application

### Spatial analysis

DBSCAN was used because it does not require the number of geographic
clusters to be specified in advance and can label isolated incidents as
noise.

Traffic incidents were matched to their nearest traffic-control point
only when the distance fell within the project's selected matching
threshold.

### Historical Incident Risk Index

The index combines:

- historical incident activity
- recent activity
- DBSCAN hotspot concentration
- adverse-weather association
- rush-hour concentration
- historical trend

The score is intentionally transparent and is not an official safety
rating.

### Predictive modelling

The model estimates hourly citywide recorded incident activity.

Training uses older calendar years and testing uses later observations.
Hours inside inferred source-collection gaps are excluded so archive outages
are not scored as genuine zero-incident periods.

### Important limitations

- Recorded incidents are not the same as true collision probability.
- Traffic-volume exposure is not included.
- Airport weather is used as a citywide proxy.
- The traffic-incidents archive can contain source-collection gaps.
- The >48-hour gap rule is a project heuristic, not an official uptime flag.
- Geographic clustering depends on project-selected parameters.
- Weather relationships are associations, not causal effects.
- The application should not be used for real-time navigation or
  individual safety decisions.
"""
    )
