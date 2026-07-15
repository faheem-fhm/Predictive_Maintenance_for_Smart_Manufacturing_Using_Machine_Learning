import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

.main{
    background-color:#0f172a;
}

.stButton>button{
    background:linear-gradient(90deg,#2563eb,#06b6d4);
    color:white;
    border:none;
    border-radius:10px;
    height:50px;
    width:100%;
    font-size:18px;
    font-weight:bold;
}

.metric-card{
    background:#1e293b;
    padding:20px;
    border-radius:15px;
    border:1px solid #334155;
    text-align:center;
}

.result-good{
    background:#14532d;
    color:white;
    padding:25px;
    border-radius:15px;
    text-align:center;
    font-size:24px;
    font-weight:bold;
}

.result-bad{
    background:#7f1d1d;
    color:white;
    padding:25px;
    border-radius:15px;
    text-align:center;
    font-size:24px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("model.pkl")

# -----------------------------
# Header
# -----------------------------
st.title("⚙ Predictive Maintenance Prediction System")
st.write("### AI Powered Machine Health Prediction")
st.write("Predict whether a machine is likely to fail based on operating conditions.")

st.divider()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Machine Parameters")

machine_type = st.sidebar.selectbox(
    "Machine Type",
    ["L", "M", "H"]
)

type_map = {
    "L":0,
    "M":1,
    "H":2
}

type_value = type_map[machine_type]

air_temp = st.sidebar.number_input(
    "Air Temperature (K)",
    250.0,
    350.0,
    300.5
)

process_temp = st.sidebar.number_input(
    "Process Temperature (K)",
    250.0,
    400.0,
    310.2
)

rpm = st.sidebar.number_input(
    "Rotational Speed (RPM)",
    500,
    5000,
    1500
)

torque = st.sidebar.number_input(
    "Torque (Nm)",
    0.0,
    100.0,
    45.0
)

tool_wear = st.sidebar.number_input(
    "Tool Wear (min)",
    0,
    300,
    120
)

# -----------------------------
# Feature Engineering
# -----------------------------
temp_difference = process_temp - air_temp
power_estimate = rpm * torque

# -----------------------------
# Layout
# -----------------------------
col1, col2 = st.columns([2,1])

with col1:

    st.subheader("Machine Inputs")

    df = pd.DataFrame({
        "Feature":[
            "Machine Type",
            "Air Temperature",
            "Process Temperature",
            "Rotational Speed",
            "Torque",
            "Tool Wear",
            "Temperature Difference",
            "Power Estimate"
        ],
        "Value":[
            machine_type,
            air_temp,
            process_temp,
            rpm,
            torque,
            tool_wear,
            round(temp_difference,2),
            round(power_estimate,2)
        ]
    })

    st.dataframe(df,use_container_width=True)

with col2:

    st.subheader("Calculated Values")

    st.metric(
        "Temperature Difference",
        f"{temp_difference:.2f} K"
    )

    st.metric(
        "Estimated Power",
        f"{power_estimate:.0f}"
    )

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🚀 Predict Machine Health"):

    input_data = pd.DataFrame([{

        "Type":type_value,
        "Air temperature [K]":air_temp,
        "Process temperature [K]":process_temp,
        "Rotational speed [rpm]":rpm,
        "Torque [Nm]":torque,
        "Tool wear [min]":tool_wear,
        "Temp_Difference":temp_difference,
        "Power_Estimate":power_estimate

    }])

    prediction = model.predict(input_data)[0]

    st.divider()

    if prediction == 0:

        st.markdown("""
        <div class="result-good">
        ✅ Machine is Healthy
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="result-bad">
        ⚠ Machine Failure Predicted
        </div>
        """, unsafe_allow_html=True)

    if hasattr(model, "predict_proba"):

        prob = model.predict_proba(input_data)[0]

        st.subheader("Prediction Confidence")

        st.progress(float(max(prob)))

        st.write(f"Healthy : **{prob[0]*100:.2f}%**")

        st.write(f"Failure : **{prob[1]*100:.2f}%**")

st.divider()

st.caption(
    "Developed using Streamlit • Predictive Maintenance AI • Production UI"
)
