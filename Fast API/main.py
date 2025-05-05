from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import List

app = FastAPI(
    title="Machine Learning API",
    description="A simple API for linear regression predictions",
    version="1.0.0"
)

class InputData(BaseModel):
    features: List[float] = Field(..., example=[6.0], description="Input features for prediction")

X_train = np.array([[1], [2], [3], [4], [5]])
y_train = np.array([2, 4, 6, 8, 10])
model = LinearRegression()
model.fit(X_train, y_train)

@app.get("/")
def read_root():
    return {
        "message": "Makine Öğrenmesi API'sine Hoş Geldiniz!",
        "endpoints": {
            "/": "Ana sayfa",
            "/predict": "POST metodu ile tahmin yapın",
            "/docs": "API dokümantasyonu",
            "/example": "Tahmin için örnek"
        }
    }

@app.post("/predict", summary="Make a prediction", description="Predicts values based on linear regression model")
async def predict(data: InputData):
    try:
        features = np.array(data.features).reshape(-1, 1)
        prediction = model.predict(features)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/predict/{feature_value}", summary="Make a prediction with GET", description="Predicts a value based on a single feature")
async def predict_get(feature_value: float):
    try:
        features = np.array([[feature_value]])
        prediction = model.predict(features)
        return {
            "feature": feature_value,
            "prediction": prediction[0]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/example", summary="Example usage")
def example():
    return {
        "message": "Make a POST request to /predict with the following JSON body:",
        "example_request": {
            "features": [6.0]
        },
        "expected_response": {
            "prediction": [12.0]  # Expected value for input feature 6
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)