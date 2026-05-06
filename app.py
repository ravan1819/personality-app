import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Personality Prediction App", layout="centered")

# ---------------- LOAD FILES ----------------
model = pickle.load(open("model.pkl", "rb"))
accuracy = pickle.load(open("accuracy.pkl", "rb"))
cm = pickle.load(open("cm.pkl", "rb"))

# ---------------- HEADER ----------------
st.title("🧠 Personality Prediction App")
st.markdown("### Discover whether you are an Introvert or Extrovert")

st.divider()

# ---------------- INPUT SECTION ----------------
st.subheader("📋 Enter Your Details")

time_alone = st.slider("Time spent alone (hrs)", 0, 24, 2)

social_events = st.slider("Social event attendance", 0, 10, 1)

going_out = st.slider("Outdoor activity frequency", 0, 10, 1)

friends = st.slider("Friends circle size", 0, 50, 5)

posts = st.slider("Social media post frequency", 0, 10, 1)

stage_fear = st.radio("Do you have stage fear?", ["Yes", "No"])

drained = st.radio("Do you feel drained after socializing?", ["Yes", "No"])

st.divider()

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Personality"):

    # Convert categorical values
    stage_fear_val = 1 if stage_fear == "Yes" else 0
    drained_val = 1 if drained == "Yes" else 0

    # Input array
    input_data = np.array([[time_alone,
                            stage_fear_val,
                            social_events,
                            going_out,
                            drained_val,
                            friends,
                            posts]])

    # Prediction
    prediction = model.predict(input_data)

    st.subheader("🎯 Prediction Result")

    if prediction[0] == 1:
        st.success("🟢 You are an Extrovert")
    else:
        st.info("🔵 You are an Introvert")

    # ---------------- ACCURACY ----------------
    st.subheader("📊 Model Accuracy")

    st.success(f"Model Accuracy: {accuracy*100:.2f}%")
    
    # ---------------- INPUT GRAPH ----------------
    st.subheader("📈 Your Input Overview")

    features = ["Alone", "Social", "Outdoor", "Friends", "Posts"]

    values = [time_alone,
              social_events,
              going_out,
              friends,
              posts]

    fig1, ax1 = plt.subplots()

    ax1.bar(features, values)

    st.pyplot(fig1)

    # ---------------- CONFUSION MATRIX ----------------
    st.subheader("📉 Confusion Matrix")

    fig2, ax2 = plt.subplots()

    sns.heatmap(cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=['Introvert', 'Extrovert'],
                yticklabels=['Introvert', 'Extrovert'])

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    st.pyplot(fig2)

# ---------------- FOOTER ----------------
st.divider()

st.markdown("Built using Machine Learning + Streamlit")
