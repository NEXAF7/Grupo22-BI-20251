import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier  # Ejemplo de modelo supervisado

def load_model(model_path):
    """Carga el modelo serializado (se espera que sea un pipeline completo)."""
    return joblib.load(model_path)

def save_model(model, model_path):
    """Guarda el modelo en la ruta indicada."""
    joblib.dump(model, model_path)

def predict(model, instances):
    """Realiza predicciones usando el pipeline que procesa texto directamente."""
    predictions = []
    for instance in instances:
        text = instance.get('text_input', '')
        if not text:
            predictions.append({'prediction': 'Not Fake', 'probability': 0.0})
            continue
        # El pipeline transforma el texto internamente (TfidfVectorizer + clasificador)
        y_pred = model.predict([text])
        probas = model.predict_proba([text])
        predictions.append({
            'prediction': int(y_pred[0]),  # Convertir a int
            'probability': float(round(max(probas[0]), 4))  # Convertir a float
        })
    return predictions

def predict_from_file(model, file_path):
    """Realiza predicciones a partir de un archivo Excel (se espera columna 'Descripción')."""
    data = pd.read_excel(file_path, engine='openpyxl')
    predictions = []
    for _, row in data.iterrows():
        text = str(row['Descripción'])
        if not text:
            predictions.append({'prediction': 'Not Fake', 'probability': 0.0})
        else:
            y_pred = model.predict([text])
            probas = model.predict_proba([text])
            predictions.append({
                'prediction': int(y_pred[0]),
                'probability': float(round(max(probas[0]), 4))
            })
    return predictions

def retrain_model(model_path, file_path):
    """
    Reentrena el modelo a partir de un archivo CSV o Excel.
    Se espera que el archivo tenga las columnas 'Título', 'Descripción' y 'Label'.
    """
    if file_path.lower().endswith('.csv'):
        # Ajusta el separador según corresponda (en este ejemplo se usa ";" si es CSV)
        data = pd.read_csv(file_path, encoding="utf-8", sep=";")
    elif file_path.lower().endswith(('.xls', '.xlsx')):
        data = pd.read_excel(file_path, engine='openpyxl')
    else:
        raise ValueError("Formato de archivo no soportado. Use CSV o Excel.")
    
    # Verificar que existan las columnas requeridas
    required_cols = ['Título', 'Descripción', 'Label']
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"El archivo debe contener la columna '{col}'.")
    
    X = data[['Título', 'Descripción']]
    y = data['Label']
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Concatenar 'Título' y 'Descripción' para formar la entrada de texto completa
    X_train_text = (X_train['Título'] + " " + X_train['Descripción']).astype(str)
    X_test_text = (X_test['Título'] + " " + X_test['Descripción']).astype(str)
    
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=10000)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    pipeline.fit(X_train_text, y_train)
    y_pred = pipeline.predict(X_test_text)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted')
    }
    return pipeline, metrics
