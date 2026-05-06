# Personality Prediction App

This project is a Machine Learning web application that predicts whether a person is an **Introvert** or **Extrovert** based on their social behavior.


📁 Project Structure

## 🚀 Features

Feature	Description
Time_spent_Alone	Average time spent alone per day
Stage_fear	Binary (Yes/No) - Fear of public speaking
Social_event_attendance	Frequency of attending social events
Going_outside	Frequency of going outside
Drained_after_socializing	Binary (Yes/No) - Feeling exhausted after socializing
Friends_circle_size	Number of close social contacts
Post_frequency	Frequency of social media posts
Personality	Target variable (Introvert = 0, Extrovert = 1)

- Predict personality using Logistic Regression
- Interactive web app using Streamlit
- User-friendly input interface
- Real-time prediction

## 📊 Input Features
- Time spent alone
- Stage fear (Yes/No)
- Social event attendance
- Going outside frequency
- Drained after socializing (Yes/No)
- Friends circle size
- Post frequency

## 🛠️ Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit

Steps Performed
✅ Handled missing values and duplicates
✅ Encoded categorical variables (Yes/No → 1/0, Introvert/Extrovert → 0/1)
✅ EDA: Histograms, KDE plots, Pair plots, Correlation Heatmaps


## ⚙️ Model
- Logistic Regression
- Data preprocessing (handling missing values & encoding)

## ▶️ How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py

## 🌐 Live Demo

[🚀 Click here to use the app](https://personality-app-mmv2hc9wbrymyqgu7pbjga.streamlit.app/)
