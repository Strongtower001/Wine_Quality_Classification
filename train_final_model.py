import pandas as pd
import joblib
from imblearn.ensemble import BalancedRandomForestClassifier

# Load dataset
df = pd.read_csv("data/winequality_categorized.csv")
# Separate features and target
X = df.drop(columns=["quality", "quality_category"])
y = df["quality_category"]

# Create final Balanced Random Forest model
model = BalancedRandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

# Train on the complete dataset
model.fit(X, y)

# Save model and important information
artifact = {
    "model": model,
    "features": X.columns.tolist(),
    "classes": model.classes_.tolist(),
    "target": "quality_category"
}

joblib.dump(artifact, "model/wine_quality_model.pkl")

print("Final model trained successfully.")
print("Training records:", len(X))
print("Number of features:", len(X.columns))
print("Features:", X.columns.tolist())
print("Classes:", model.classes_.tolist())
print("Model saved as: wine_quality_model.pkl")