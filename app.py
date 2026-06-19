import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="",
    layout="wide"
)

st.markdown("""
<style>
.main{
    padding-top:1rem;
}
.metric-card{
    background-color:#f0f2f6;
    padding:15px;
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

st.title("Predictive Maintenance Prediction System")
st.markdown("### Predict Machine Failure using Machine Learning")

@st.cache_resource
def load_model():
    with open("model.pkl","rb") as file:
        return pickle.load(file)

model = load_model()

col1, col2 = st.columns(2)

with col1:
    machine_type = st.selectbox("Machine Type", ["L","M","H"])
    air_temp = st.number_input("Air Temperature (K)", value=300.0)
    process_temp = st.number_input("Process Temperature (K)", value=310.0)
    rotational_speed = st.number_input("Rotational Speed (rpm)", value=1500)

with col2:
    torque = st.number_input("Torque (Nm)", value=45.0)
    tool_wear = st.number_input("Tool Wear (min)", value=120)
    
type_map = {"L":0,"M":1,"H":2}
machine_type_encoded = type_map[machine_type]

temp_difference = process_temp - air_temp
power_estimate = rotational_speed * torque

st.subheader("Derived Features")

c1, c2 = st.columns(2)
c1.metric("Temperature Difference", f"{temp_difference:.2f}")
c2.metric("Power Estimate", f"{power_estimate:.0f}")

if st.button("🔍 Predict Machine Status", use_container_width=True):

    input_data = pd.DataFrame([{
        "Type": machine_type_encoded,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "Temp_Difference": temp_difference,
        "Power_Estimate": power_estimate
    }])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("Machine Failure Predicted")
    else:
        st.success("Machine is Healthy")

st.markdown("---")
st.markdown("Developed by **Mohamed Faheem** | Streamlit Dashboard")
