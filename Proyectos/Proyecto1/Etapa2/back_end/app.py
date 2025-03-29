from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
import traceback
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

app = Flask(__name__, 
            template_folder='../front_end/templates', 
            static_folder='../front_end/static')

MODEL_PATH = 'model/model.pkl'
ALLOWED_EXTENSIONS = {'xlsx', 'csv'}
app.config['UPLOAD_FOLDER'] = 'temp'

model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model(file_path=None):
    file_path = file_path or MODEL_PATH
    if os.path.exists(file_path):
        loaded_model = joblib.load(file_path)
        if not isinstance(loaded_model, Pipeline):
            raise ValueError("El modelo no es un pipeline válido. Verifica el proceso de entrenamiento.")
        return loaded_model
    raise FileNotFoundError("Modelo no encontrado en la ruta: {}".format(file_path))

def save_model(model_object, file_path=None):
    file_path = file_path or MODEL_PATH
    joblib.dump(model_object, file_path)

def create_pipeline():
    return Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=10000)),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

@app.route('/')
def index():
    """Renderiza la interfaz principal."""
    return render_template('index.html')

@app.route('/load_model', methods=['POST'])
def load_model_endpoint():
    """Endpoint para cargar archivo de modelo."""
    global model
    model_file = request.files.get('model_file')

    if model_file and model_file.filename.endswith('.pkl'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(model_file.filename))
        model_file.save(file_path)
        try:
            model = load_model(file_path)
            return jsonify({'message': 'Modelo cargado exitosamente'}), 200
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Debe proporcionar un archivo .pkl válido'}), 400

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint para realizar predicciones.
    Si se provee el archivo en el campo 'opinion_file', se procesa el archivo.
    Si no, se toma el texto del campo 'Descripcion'.
    """
    global model
    if model is None:
        return jsonify({'error': 'El modelo no está cargado. Por favor, cargue el modelo primero.'}), 400

    try:
        # Si se envía un archivo, se procesa la predicción en lote
        if 'opinion_file' in request.files and request.files.get('opinion_file').filename != "":
            file = request.files.get('opinion_file')
            if not allowed_file(file.filename):
                return jsonify({'error': 'Debe proporcionar un archivo CSV/XLSX válido'}), 400

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(file_path)
            data = pd.read_excel(file_path) if file_path.endswith(".xlsx") else pd.read_csv(file_path, sep=';')

            if "Descripcion" not in data.columns:
                return jsonify({'error': 'El archivo debe contener la columna "Descripcion"'}), 400

            X = data["Descripcion"].tolist()
            preds = model.predict(X)
            probs = model.predict_proba(X).tolist()

            results = []
            for i in range(len(X)):
                results.append({
                    'Descripcion': X[i],
                    'prediction': int(preds[i]),
                    'probabilities': probs[i]
                })

            return jsonify({'predictions': results}), 200

        else:
            # Si no se envía archivo, se procesa el campo de texto
            descripcion = request.form.get('Descripcion')
            if not descripcion:
                return jsonify({'error': 'Debe proporcionar una Descripcion válida'}), 400

            X = [descripcion]
            pred_class = model.predict(X)
            pred_proba = model.predict_proba(X).tolist()
            predictions = {
                'Descripcion': descripcion,
                'prediction': int(pred_class[0]),
                'probabilities': pred_proba[0]
            }
            return jsonify({'predictions': predictions}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    """Endpoint para reentrenar el modelo con archivo etiquetado."""
    global model
    if model is None:
        return jsonify({'error': 'El modelo no está cargado. Por favor, cargue el modelo primero.'}), 400

    try:
        labels_file = request.files.get('labels_file')
        if labels_file and allowed_file(labels_file.filename):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(labels_file.filename))
            labels_file.save(file_path)
            # Se lee el archivo Excel o CSV según la extensión
            data = pd.read_excel(file_path) if file_path.endswith(".xlsx") else pd.read_csv(file_path, sep=';')
            
            # Se utilizan las columnas 'Descripcion' y 'label' como variables de interés
            if "Descripcion" not in data.columns or "label" not in data.columns:
                return jsonify({'error': 'El archivo debe contener las columnas "Descripcion" y "label"'}), 400

            X = data["Descripcion"]
            y = data["label"]

            pipeline = create_pipeline()
            pipeline.fit(X, y)

            save_model(pipeline)
            model = pipeline

            y_pred = pipeline.predict(X)
            precision = precision_score(y, y_pred, average='weighted')
            recall = recall_score(y, y_pred, average='weighted')
            f1 = f1_score(y, y_pred, average='weighted')

            return jsonify({
                'message': 'Modelo reentrenado exitosamente.',
                'metrics': {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }
            })
        else:
            return jsonify({'error': 'Debe proporcionar un archivo válido para reentrenar'}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
