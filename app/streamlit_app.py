import streamlit as st

st.set_page_config(page_title="VLE predictor")
st.title(" Vapor Pressure Prediction from SMILES and temperature")
smiles_string = st.text_input("Enter the SMILES: ", placeholder="e.g. CCO for ethanol")
temp_input = st.number_input("Enter temperature in K: ", min_value = 80.0, max_value = 1000.0, value = 298.15, step = 1.0)

# Check for an empty string
if not smiles_string:
    st.info("Enter a SMILES string to start.")
    st.stop()

st.write("SMILES String", smiles_string)
st.write("Temperature in K", temp_input)