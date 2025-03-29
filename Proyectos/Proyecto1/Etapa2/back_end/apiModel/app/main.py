from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
import os
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import json
import logging
from datetime import datetime

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

MODEL_PATH = "model/model.pkl"

# Esquema para solicitudes de predicción
class PredictionRequest(BaseModel):
    text: str

# Esquema para solicitudes de reentrenamiento
class RetrainRequest(BaseModel):
    data: List[str]  # Cada entrada es un texto (por ejemplo, concatenación de Título y Descripción)
    target: List[int]

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    raise HTTPException(status_code=404, detail="Modelo no encontrado")

def save_model(model):
    joblib.dump(model, MODEL_PATH)

@app.get("/")
def read_root():
    return {"message": "API de Evaluación de Noticias"}

@app.post("/predict")
async def predict_endpoint(request: PredictionRequest):
    model = load_model()
    try:
        y_pred = model.predict([request.text])
        probabilities = model.predict_proba([request.text]).max(axis=1).tolist()
        # Convertir valores a tipos nativos (int y float) para JSON
        return {"prediction": int(y_pred[0]), "probability": float(probabilities[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/retrain")
async def retrain_endpoint(request: RetrainRequest):
    try:
        X = pd.DataFrame(request.data, columns=["text"])
        y = request.target

        pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(max_features=10000)),
            ('classifier', RandomForestClassifier(n_estimators=20, random_state=42))
        ])

        pipeline.fit(X["text"], y)
        save_model(pipeline)

        y_pred = pipeline.predict(X["text"])
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
def retraining_info():
    return {
        "strategies": RETRAINING_STRATEGIES,
        "implemented_strategy": IMPLEMENTED_STRATEGY
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
