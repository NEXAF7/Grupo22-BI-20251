import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier  # Ejemplo de modelo supervisado

# Función para cargar modelo serializado
def load_model(model_path):
    return joblib.load(model_path)

# Función para guardar modelo actualizado
def save_model(model, model_path):
    joblib.dump(model, model_path)

# Función para predicción basada en JSON
def predict(model, instances):
    predictions = []
    for instance in instances:
        # Simulación: Convertir propiedades a formato adecuado para el modelo
        fake_prediction = {'prediction': 'Fake' if instance['text_input'] else 'Not Fake', 'probability': 0.85}  # Ejemplo simple
        predictions.append(fake_prediction)
    return predictions

# Función para predicción desde archivo
def predict_from_file(model, file_path):
    # Leer el archivo y hacer predicciones
    data = pd.read_excel(file_path)  # Asegúrate de usar un archivo Excel válido
    predictions = [{'prediction': 'Fake' if row['Label'] == 'Fake' else 'Not Fake', 'probability': 0.85} for _, row in data.iterrows()]
    return predictions

# Función para reentrenar el modelo
def retrain_model(model_path, file_path):
    data = pd.read_excel(file_path)  # Leer datos etiquetados desde archivo
    X = data[['Título', 'Descripción']]  # Características
    y = data['Label']  # Etiqueta

    # Preprocesar y dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Reentrenar modelo
    new_model = RandomForestClassifier()
    new_model.fit(X_train, y_train)
    y_pred = new_model.predict(X_test)

    # Generar nuevas métricas
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted'),
    }
    return new_model, metrics