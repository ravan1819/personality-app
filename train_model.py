import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

# Load dataset
data = pd.read_csv("personality_dataset.csv")

# Clean
data = data.dropna().drop_duplicates()

# Encode
data['Stage_fear'] = data['Stage_fear'].map({'Yes':1, 'No':0})
data['Drained_after_socializing'] = data['Drained_after_socializing'].map({'Yes':1, 'No':0})
data['Personality'] = data['Personality'].map({'Extrovert':1, 'Introvert':0})

# Features & target
X = data.drop("Personality", axis=1)
y = data["Personality"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

# Save metrics
pickle.dump(accuracy, open("accuracy.pkl", "wb"))
pickle.dump(cm, open("cm.pkl", "wb"))

print("Model and metrics saved!")
