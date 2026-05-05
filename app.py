import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Title
st.title("🧠 Personality Prediction App")
st.markdown("### Predict whether a person is Introvert or Extrovert")

# Sidebar
st.sidebar.title("About")
st.sidebar.info("This app predicts personality using Machine Learning (Logistic Regression).")

st.write("Fill the details below:")

# Inputs
time_alone = st.number_input("Time spent alone", min_value=0, value=2)

stage_fear = st.selectbox("Stage Fear", ["Yes", "No"])

social_events = st.number_input("Social event attendance", min_value=0, value=1)

going_out = st.number_input("Outdoor activity frequency", min_value=0, value=1)

drained = st.selectbox("Drained after socializing", ["Yes", "No"])

friends = st.number_input("Friends circle size", min_value=0, value=5)

posts = st.number_input("Post frequency", min_value=0, value=1)

# Convert categorical
stage_fear = 1 if stage_fear == "Yes" else 0
drained = 1 if drained == "Yes" else 0

# Predict
if st.button("Predict"):
    input_data = np.array([[time_alone, stage_fear, social_events, going_out, drained, friends, posts]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("🟢 Extrovert")
    else:
        st.error("🔵 Introvert")

    st.info("This prediction is based on a Logistic Regression model trained on social behavior data.")
