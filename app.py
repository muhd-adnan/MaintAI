import streamlit as st
import pandas as pd
import joblib
import textwrap
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="MaintAI - Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)

# ---------------------------------------------------------
# MaintAI dark, minimal interface
# ---------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #111214;
        color: #F5F5F7;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #F5F5F7 !important;
        letter-spacing: -0.5px;
    }

    p, label {
        color: #D1D1D6;
    }

    label {
        color: #A1A1AA !important;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background: #1C1C1E !important;
        border: 1px solid #38383A !important;
        border-radius: 12px !important;
    }

    div.stButton > button {
        background: #F5F5F7 !important;
        color: #111214 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 1rem;
        font-weight: 600;
        transition: background 0.2s ease;
    }

    div.stButton > button p {
        color: #111214 !important;
        font-weight: 600 !important;
    }

    div.stButton > button:hover {
        background: #D1D1D6 !important;
        border: none !important;
    }

    hr {
        border-color: #38383A;
    }

    .main-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 25px;
        border-bottom: 1px solid #29292C;
        margin-bottom: 52px;
    }

    .brand {
        font-size: 27px;
        font-weight: 700;
        letter-spacing: -1.2px;
        color: #F5F5F7;
    }

    .brand-dot {
        color: #30D158;
    }

    .system-status {
        font-size: 10px;
        letter-spacing: 1.5px;
        color: #A1A1AA;
        display: flex;
        align-items: center;
        gap: 9px;
    }

    .status-indicator {
        width: 7px;
        height: 7px;
        background: #30D158;
        border-radius: 50%;
        display: inline-block;
    }

    .hero-section {
        margin-bottom: 34px;
    }

    .hero-label {
        color: #30D158;
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 2px;
        margin-bottom: 15px;
    }

    .hero-section h1 {
        font-size: 46px;
        line-height: 1.12;
        letter-spacing: -2px;
        font-weight: 650;
        margin: 0 0 16px 0;
        color: #F5F5F7;
    }

    .hero-section p {
        color: #A1A1AA;
        font-size: 15px;
        max-width: 570px;
        line-height: 1.7;
    }

    .section-caption {
        color: #A1A1AA;
        font-size: 13px;
    }

    @media (max-width: 700px) {
        .hero-section h1 {
            font-size: 35px;
        }

        .main-header {
            margin-bottom: 36px;
        }

        .system-status {
            font-size: 8px;
            letter-spacing: 1px;
        }
    }
    
    /* Sensor summary cards */
    .sensor-card {
        background: #1C1C1E;
        border: 1px solid #38383A;
        border-radius: 14px;
        padding: 20px;
        min-height: 85px;
        box-sizing: border-box;
    }

    .sensor-label {
        color: #A1A1AA;
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 1.2px;
        margin-bottom: 13px;
    }

    .sensor-value {
        color: #F5F5F7;
        font-size: 23px;
        font-weight: 600;
        letter-spacing: -0.5px;
    }

    .sensor-value span {
        color: #A1A1AA;
        font-size: 13px;
        font-weight: 400;
    }

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load the saved machine learning pipeline
# ---------------------------------------------------------
@st.cache_resource
def load_pipeline():
    model_path = Path(__file__).parent / "maintai_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            "maintai_model.joblib was not found in the application folder."
        )

    return joblib.load(model_path)


try:
    pipeline = load_pipeline()
    preprocessor = pipeline["preprocessor"]
    model = pipeline["model"]
    threshold = float(pipeline["threshold"])
    selected_features = pipeline["features"]

except Exception as error:
    st.error(f"Unable to load the machine learning model: {error}")
    st.stop()

# ---------------------------------------------------------
# Dashboard header
# ---------------------------------------------------------
st.markdown("""
<div class="main-header">
    <div class="brand">
        MaintAI<span class="brand-dot">.</span>
    </div>
    <div class="system-status">
        <span class="status-indicator"></span>
        PREDICTIVE SYSTEM
    </div>
</div>

<div class="hero-section">
    <div class="hero-label">INTELLIGENT MACHINE MONITORING</div>
    <h1>Machine health,<br>made intelligent.</h1>
    <p>
        Analyse machine sensor readings and identify
        conditions associated with possible failure.
    </p>
</div>
""", unsafe_allow_html=True)

st.caption(
    "MaintAI classifies current sensor conditions. "
    "It does not predict an exact future failure date or time."
)

# ---------------------------------------------------------
# Sensor input section
# ---------------------------------------------------------
st.markdown("## Machine parameters")
st.caption("Configure the sensor readings for machine condition analysis.")

col1, col2 = st.columns(2)

with col1:
    machine_type = st.selectbox(
        "Machine Type",
        options=["L", "M", "H"]
    )

    air_temperature = st.number_input(
        "Air Temperature (K)",
        min_value=295.3,
        max_value=304.5,
        value=300.0,
        step=0.1
    )

    process_temperature = st.number_input(
        "Process Temperature (K)",
        min_value=305.7,
        max_value=313.8,
        value=310.0,
        step=0.1
    )

with col2:
    rotational_speed = st.number_input(
        "Rotational Speed (rpm)",
        min_value=1168,
        max_value=2886,
        value=1500
    )

    torque = st.number_input(
        "Torque (Nm)",
        min_value=3.8,
        max_value=76.6,
        value=40.0,
        step=0.1
    )

    tool_wear = st.number_input(
        "Tool Wear (min)",
        min_value=0,
        max_value=253,
        value=100
    )

st.divider()

# Initialize analysis history
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []


# Sensor input validation

validation_errors = []

if rotational_speed < 0:
    validation_errors.append("Rotational speed cannot be negative.")

if torque < 0:
    validation_errors.append("Torque cannot be negative.")

if tool_wear < 0:
    validation_errors.append("Tool wear cannot be negative.")

if process_temperature <= air_temperature:
    validation_errors.append(
        "Process temperature should be higher than air temperature."
    )

if validation_errors:
    for error in validation_errors:
        st.warning(error)


# Check readings against the training dataset ranges

range_warnings = []

sensor_ranges = {
    "Air temperature": (air_temperature, 295.3, 304.5, "K"),
    "Process temperature": (process_temperature, 305.7, 313.8, "K"),
    "Rotational speed": (rotational_speed, 1168, 2886, "rpm"),
    "Torque": (torque, 3.8, 76.6, "Nm"),
    "Tool wear": (tool_wear, 0, 253, "min")
}

for sensor, (value, minimum, maximum, unit) in sensor_ranges.items():
    if value < minimum or value > maximum:
        range_warnings.append(
            f"{sensor} is outside the dataset range "
            f"({minimum}–{maximum} {unit})."
        )

if range_warnings:
    st.warning(
        "Some readings are outside the ranges observed "
        "in the training dataset:"
    )

    for warning in range_warnings:
        st.write(f"- {warning}")

# ---------------------------------------------------------
# Prediction and result display
# ---------------------------------------------------------
if st.button(
    "Analyse Machine Condition",
    type="primary",
    use_container_width=True
) and not validation_errors:
    
    input_data = pd.DataFrame([{
        "Type": machine_type,
        "Air temperature [K]": air_temperature,
        "Process temperature [K]": process_temperature,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear
    }])

    # Apply the same preprocessing used during model training.
    processed_data = preprocessor.transform(input_data)

    # Calculate the probability of machine failure.
    failure_probability = float(
        model.predict_proba(processed_data)[0][1]
    )

    # Apply the saved classification threshold.
    prediction = int(failure_probability >= threshold)

    probability_percent = failure_probability * 100
    probability_width = max(0.0, min(100.0, probability_percent))

    if prediction == 1:
        status = "Potential Failure Detected"
        status_color = "#FF453A"
        status_message = (
            "The model identified a potential failure condition. "
            "Inspect and verify the machine's operating condition."
        )
        status_icon = "!"
    else:
        status = "No Failure Detected"
        status_color = "#30D158"
        status_message = (
            "The model classified these readings as a "
            "non-failure condition."
        )
        status_icon = "✓"
        # Save this analysis
    st.session_state.analysis_history.append({
        "Analysis": len(st.session_state.analysis_history) + 1,
        "Date & Time": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "Machine Type": machine_type,
        "Condition": status,
        "Failure Probability": f"{probability_percent:.2f}%"
    })

    st.markdown("### Machine condition")
    st.caption("ANALYSIS COMPLETE")

    # st.html renders this as HTML, rather than displaying the markup as code.
    result_html = f"""
    <div style="
        background: #1C1C1E;
        border: 1px solid #38383A;
        border-radius: 18px;
        padding: 26px;
        margin-top: 12px;
        margin-bottom: 15px;
        color: #F5F5F7;
        font-family: sans-serif;
    ">
        <div style="
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 15px;
            flex-wrap: wrap;
        ">
            <div>
                <div style="
                    font-size: 12px;
                    color: #A1A1AA;
                    letter-spacing: 1px;
                ">
                    PREDICTED STATUS
                </div>

                <div style="
                    font-size: 25px;
                    font-weight: 650;
                    color: {status_color};
                    margin-top: 9px;
                ">
                    {status}
                </div>
            </div>

            <div style="
                width: 48px;
                height: 48px;
                border-radius: 50%;
                background: {status_color}18;
                color: {status_color};
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 25px;
                font-weight: 600;
            ">
                {status_icon}
            </div>
        </div>

        <div style="
            margin-top: 28px;
            font-size: 12px;
            color: #A1A1AA;
            letter-spacing: 1px;
        ">
            FAILURE PROBABILITY
        </div>

        <div style="
            font-size: 42px;
            font-weight: 650;
            letter-spacing: -2px;
            color: #F5F5F7;
            margin-top: 4px;
        ">
            {probability_percent:.2f}%
        </div>

        <div style="
            height: 6px;
            background: #38383A;
            border-radius: 10px;
            margin-top: 17px;
            overflow: hidden;
        ">
            <div style="
                width: {probability_width:.2f}%;
                height: 100%;
                background: {status_color};
                border-radius: 10px;
            "></div>
        </div>

        <div style="
            display: flex;
            justify-content: space-between;
            margin-top: 12px;
            font-size: 11px;
            color: #A1A1AA;
        ">
            <span>0%</span>
            <span>100%</span>
        </div>
    </div>
    """

    st.html(result_html)

    st.markdown(
        f"<p style='color:#A1A1AA;font-size:14px;'>{status_message}</p>",
        unsafe_allow_html=True
    )

    st.caption(
        f"Classification threshold: {threshold:.0%}"
    )

    # Sensor readings summary

    st.markdown("### Sensor readings")
    st.caption("Input parameters used for this analysis.")

    sensor_summary_html = f"""
    <div style="
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px;
        margin-top: 16px;
    ">
        <div class="sensor-card">
            <div class="sensor-label">MACHINE TYPE</div>
            <div class="sensor-value">{machine_type}</div>
        </div>

        <div class="sensor-card">
            <div class="sensor-label">AIR TEMPERATURE</div>
            <div class="sensor-value">{air_temperature:.1f} <span>K</span></div>
        </div>

        <div class="sensor-card">
            <div class="sensor-label">PROCESS TEMPERATURE</div>
            <div class="sensor-value">{process_temperature:.1f} <span>K</span></div>
        </div>

        <div class="sensor-card">
            <div class="sensor-label">ROTATIONAL SPEED</div>
            <div class="sensor-value">{rotational_speed} <span>rpm</span></div>
        </div>

        <div class="sensor-card">
            <div class="sensor-label">TORQUE</div>
            <div class="sensor-value">{torque:.1f} <span>Nm</span></div>
        </div>

        <div class="sensor-card">
            <div class="sensor-label">TOOL WEAR</div>
            <div class="sensor-value">{tool_wear} <span>min</span></div>
        </div>
    </div>
    """

    st.html(textwrap.dedent(sensor_summary_html))


# Recent analysis history

st.divider()

st.markdown("## Recent analyses")
st.caption("Your latest machine condition assessments.")

if st.session_state.analysis_history:

    history_df = pd.DataFrame(
        st.session_state.analysis_history[::-1]
    )

    st.dataframe(
        history_df,
        hide_index=True,
        use_container_width=True
    )

    if st.button("Clear history"):
        st.session_state.analysis_history = []
        st.rerun()

else:
    st.markdown("""
    <div style="
        background:#1C1C1E;
        border:1px solid #38383A;
        border-radius:14px;
        padding:24px;
        color:#A1A1AA;
        text-align:center;
    ">
        No analyses yet. Run your first machine assessment.
    </div>
    """, unsafe_allow_html=True)

# Model information

st.divider()

st.markdown("## Model information")
st.caption(
    "Machine learning architecture and evaluation performance."
)

# Model details
model_col1, model_col2 = st.columns(2)

with model_col1:
    st.markdown("##### MODEL ARCHITECTURE")
    st.markdown("Random Forest Classifier")

with model_col2:
    st.markdown("##### CLASSIFICATION THRESHOLD")
    st.markdown("40%")

st.markdown("### Evaluation performance")
st.caption("Results obtained from the held-out test dataset.")

# Performance metrics
metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    st.metric("Accuracy", "98.30%")

with metric_col2:
    st.metric("Precision", "85.42%")

with metric_col3:
    st.metric("Recall", "60.29%")

metric_col4, metric_col5 = st.columns(2)

with metric_col4:
    st.metric("F1-Score", "70.69%")

with metric_col5:
    st.metric("PR-AUC", "76.42%")

st.caption(
    "Evaluation was performed using the AI4I 2020 synthetic "
    "predictive maintenance dataset. Results on synthetic data "
    "do not guarantee equivalent performance in real factories."
)
