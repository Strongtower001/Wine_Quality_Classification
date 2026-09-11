from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load the trained model
artifact = joblib.load("../model/wine_quality_model.pkl")

model = artifact["model"]
features = artifact["features"]
classes = artifact["classes"]

# Create FastAPI application
app = FastAPI(
    title="Wine Quality Classification API",
    description="API for predicting wine quality category using Balanced Random Forest",
    version="1.0.0"
)


# Input data structure
class WineData(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float


@app.get("/")
def home():
    return {
        "message": "Wine Quality Classification API is running",
        "classes": classes
    }


@app.post("/predict")
def predict(data: WineData):

    input_data = pd.DataFrame([[
        data.fixed_acidity,
        data.volatile_acidity,
        data.citric_acid,
        data.residual_sugar,
        data.chlorides,
        data.free_sulfur_dioxide,
        data.total_sulfur_dioxide,
        data.density,
        data.pH,
        data.sulphates,
        data.alcohol
    ]], columns=features)

    prediction = model.predict(input_data)[0]

    return {
        "predicted_quality": prediction,
        "classes": classes
    }