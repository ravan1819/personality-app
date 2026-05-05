import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
data = pd.read_csv("personality_dataset.csv")

# Clean data
data = data.dropna().drop_duplicates()

# Encode categorical columns
data['Stage_fear'] = data['Stage_fear'].map({'Yes':1, 'No':0})
data['Drained_after_socializing'] = data['Drained_after_socializing'].map({'Yes':1, 'No':0})
data['Personality'] = data['Personality'].map({'Extrovert':1, 'Introvert':0})

X = data.drop("Personality", axis=1)
y = data["Personality"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model saved successfully!")