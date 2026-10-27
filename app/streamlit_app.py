import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="VLE predictor")

import sys
from pathlib import Path

import joblib
import streamlit as st
from xgboost import XGBRegressor

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]

sys.path.append(str(ROOT))

from src.predict import predict_from_smiles


@st.cache_resource
def load_model_and_scaler():
    """Load the tuned XGBoost model and the temperature scaler (once)."""
    print("loading model")
    model = XGBRegressor()
    model.load_model(ROOT / "models" / "models" / "xgb_tuned.json")
    scaler = joblib.load(ROOT / "models" / "temperature_scaler.joblib")
    return model, scaler


model, scaler = load_model_and_scaler()
st.title(" Vapor Pressure Prediction from SMILES and temperature")
col1, col2 = st.columns(2)
with col1:
    smiles_string = st.text_input("Enter the SMILES: ", placeholder="e.g. CCO for ethanol")
with col2:    
    temp_input = st.number_input("Enter temperature in K: ", min_value = 80.0, max_value = 1000.0, value = 298.15, step = 1.0)

# Check for an empty string
if not smiles_string:
    st.info("Enter a SMILES string to start.")
    st.stop()
try:
    result = predict_from_smiles(smiles_string, temp_input, model, scaler, min_bits=1)
except ValueError as e:
    st.error(str(e))
    st.stop()


st.write("SMILES String", smiles_string)
st.write("Temperature in K", temp_input)
#st.write("Result:", result)
st.metric(label="Vapor pressure (kPa)", value=f"{result['pressure_kPa']:.2f}")

if result['warn']:
    st.warning("The molecule is too small to predict. The result may not be reliable")

# Drawing a temp_pressure profile
t_min, t_max = st.slider(
    "Temperature range (K)",
    min_value=80.0, max_value=1000.0,
    value=(250.0, 400.0), step=5.0,
)
# st.write(t_min, t_max)    
temps = np.linspace(t_min, t_max, 50)
pressures = []
for temp in temps:
    result_temp = predict_from_smiles(smiles_string, temp, model, scaler, min_bits=1)
    pressures.append(result_temp["pressure_kPa"])
# st.write(len(pressures))  

# Plotting the graph
df = pd.DataFrame({"Temperature (K)": temps, "Pressure (kPa)": pressures})
st.subheader(f"Vapor pressure curve for {smiles_string.strip()}")
st.line_chart(df, x="Temperature (K)", y="Pressure (kPa)")