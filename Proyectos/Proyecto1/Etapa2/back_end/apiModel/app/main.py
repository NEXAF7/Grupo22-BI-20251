from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
import os
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

MODEL_PATH = "model/model.pkl"

class PredictionRequest(BaseModel):
    text: str

class RetrainRequest(BaseModel):
    data: List[str]
    target: List[int]

def load_model():
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        if not isinstance(model, Pipeline):
            logger.error("El modelo cargado no es un Pipeline válido.")
            raise HTTPException(
                status_code=500,
                detail="El modelo cargado no es un Pipeline válido. Asegúrate de haber entrenado y guardado un pipeline."
            )
        logger.info(f"Modelo cargado exitosamente desde {MODEL_PATH}.")
        return model
    logger.error(f"Modelo no encontrado en {MODEL_PATH}.")
    raise HTTPException(status_code=404, detail="Modelo no encontrado")

def save_model(model):
    try:
        joblib.dump(model, MODEL_PATH)
        logger.info(f"Modelo guardado correctamente en {MODEL_PATH}.")
    except Exception as e:
        logger.error(f"Error al guardar el modelo: {e}")
        raise HTTPException(status_code=500, detail="Error al guardar el modelo.")

def create_pipeline():
    return Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=10000)),
        ('classifier', RandomForestClassifier(n_estimators=20, random_state=42))
    ])

@app.get("/")
def read_root():
    return {"message": "API de Evaluación de Noticias"}

@app.post("/predict")
async def predict_endpoint(request: PredictionRequest):
    model = load_model()
    try:
        y_pred = model.predict([request.text])
        probabilities = model.predict_proba([request.text])[0] 
        max_probability = max(probabilities) 
        return {
            "prediction": int(y_pred[0]),
            "probability": float(max_probability),
            "class_probabilities": probabilities.tolist()  
        }
    except Exception as e:
        logger.error(f"Error en predict_endpoint: {e}")
        raise HTTPException(status_code=400, detail=f"Error en la predicción: {str(e)}")

@app.post("/retrain")
async def retrain_endpoint(request: RetrainRequest):
    try:
        if len(request.data) != len(request.target):
            logger.error("El tamaño de los datos y las etiquetas no coincide.")
            raise HTTPException(status_code=400, detail="El tamaño de los datos y las etiquetas no coincide.")

        X = pd.DataFrame(request.data, columns=["text"])
        y = request.target

        pipeline = create_pipeline()
        pipeline.fit(X["text"], y)
        save_model(pipeline)

        y_pred = pipeline.predict(X["text"])
        precision = precision_score(y, y_pred, average='weighted')
        recall = recall_score(y, y_pred, average='weighted')
        f1 = f1_score(y, y_pred, average='weighted')

        logger.info("Modelo reentrenado exitosamente.")
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
    except Exception as e:
        logger.error(f"Error en retrain_endpoint: {e}")
        raise HTTPException(status_code=400, detail=f"Error al reentrenar el modelo: {str(e)}")

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