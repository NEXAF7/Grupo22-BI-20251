from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import os

app = FastAPI()

MODEL_PATH = "app/model/model.pkl"

class PredictionRequest(BaseModel):
    data: List[List[float]]

class RetrainRequest(BaseModel):
    data: List[List[float]]
    target: List[int]

def create_pipeline():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier())
    ])
    return pipeline

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return create_pipeline()

def save_model(model):
    joblib.dump(model, MODEL_PATH)

@app.post("/predict")
async def predict(request: PredictionRequest):
    model = load_model()
    try:
        X = pd.DataFrame(request.data)
        predictions = model.predict(X).tolist()
        probabilities = model.predict_proba(X).max(axis=1).tolist()
        return {"predictions": predictions, "probabilities": probabilities}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/retrain")
async def retrain(request: RetrainRequest):
    try:
        X = pd.DataFrame(request.data)
        y = request.target
        model = create_pipeline()
        model.fit(X, y)
        save_model(model)

        y_pred = model.predict(X)
        precision = precision_score(y, y_pred, average='weighted')
        recall = recall_score(y, y_pred, average='weighted')
        f1 = f1_score(y, y_pred, average='weighted')

        return {"precision": precision, "recall": recall, "f1_score": f1}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

RETRAINING_STRATEGIES = [
    {
        "name": "Reentrenamiento completo",
        "description": "Entrenar el modelo desde cero utilizando todos los datos disponibles (anteriores + nuevos)",
        "advantage": "Permite que el modelo aproveche toda la información disponible.",
        "disadvantage": "Consume mucho tiempo y recursos."
    },
    {
        "name": "Reentrenamiento incremental",
        "description": "Actualizar el modelo añadiendo solo los nuevos datos sin olvidar el conocimiento previo.",
        "advantage": "Más eficiente en tiempo y recursos.",
        "disadvantage": "Puede generar sesgos si los datos nuevos son muy diferentes."
    },
    {
        "name": "Transfer Learning",
        "description": "Utilizar un modelo previamente entrenado y ajustar solo algunas capas o parámetros con nuevos datos.",
        "advantage": "Reduce la cantidad de datos necesarios y mejora el desempeño.",
        "disadvantage": "No siempre es aplicable a cualquier tipo de modelo."
    }
]

IMPLEMENTED_STRATEGY = "Reentrenamiento completo"

@app.get("/retraining_info")
async def retraining_info():
    return {
        "strategies": RETRAINING_STRATEGIES,
        "implemented_strategy": IMPLEMENTED_STRATEGY
    }
