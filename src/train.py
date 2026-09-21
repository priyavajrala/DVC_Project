import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load dataset
df = pd.read_csv("data/iris.csv")

# Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Random Forest Model (Version 2)
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluation Metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted')
recall = recall_score(y_test, predictions, average='weighted')
f1 = f1_score(y_test, predictions, average='weighted')

# Display Results
print("========== Random Forest Model Version 2 ==========")
print("Number of Trees :", model.n_estimators)
print("Maximum Depth   :", model.max_depth)
print("Accuracy        :", round(accuracy, 4))
print("Precision       :", round(precision, 4))
print("Recall          :", round(recall, 4))
print("F1-Score        :", round(f1, 4))

# Save model
joblib.dump(model, "models/random_forest_v2.pkl")

print("\nModel saved successfully!")