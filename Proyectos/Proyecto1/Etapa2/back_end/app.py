from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
import traceback
import model.model_utils as model_utils  # Funciones para cargar/reentrenar el modelo

app = Flask(__name__)

# Ruta para archivo del modelo
MODEL_PATH = 'model/model.pkl'
ALLOWED_EXTENSIONS = {'xlsx', 'csv'}
app.config['UPLOAD_FOLDER'] = 'temp'

# Inicializamos el modelo en `None`
model = None

# Función para verificar extensiones de archivo permitidos
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Renderiza la interfaz principal."""
    return render_template('index.html')


@app.route('/load_model', methods=['POST'])
def load_model():
    """Endpoint para cargar un archivo de modelo."""
    global model  # Declaración global de la variable model
    model_file = request.files.get('model_file')

    if model_file and model_file.filename.endswith('.pkl'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(model_file.filename))
        model_file.save(file_path)
        try:
            model = model_utils.load_model(file_path)
            return jsonify({'message': 'Modelo cargado exitosamente'}), 200
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Debe proporcionar un archivo .pkl válido'}), 400


@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint para realizar predicciones."""
    global model  # Declaración global de model
    if model is None:
        return jsonify({'error': 'El modelo no está cargado. Por favor, cargue el modelo primero.'}), 400

    try:
        text = request.form.get('text')  # Entrada de texto
        opinion_file = request.files.get('opinion_file')  # Entrada de archivo
        predictions = {}

        if text:
            data_instance = {'text_input': text}
            predictions = model_utils.predict(model, [data_instance])
        elif opinion_file and allowed_file(opinion_file.filename):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(opinion_file.filename))
            opinion_file.save(file_path)
            predictions = model_utils.predict_from_file(model, file_path)
        else:
            return jsonify({'error': 'Debe proporcionar un texto o un archivo válido'}), 400

        return jsonify({'predictions': predictions})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/retrain', methods=['POST'])
def retrain():
    """Endpoint para reentrenar el modelo."""
    global model  # Declaración global de model
    if model is None:
        return jsonify({'error': 'El modelo no está cargado. Por favor, cargue el modelo primero.'}), 400

    try:
        labels_file = request.files.get('labels_file')

        if labels_file and allowed_file(labels_file.filename):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(labels_file.filename))
            labels_file.save(file_path)
            new_model, metrics = model_utils.retrain_model(MODEL_PATH, file_path)
            model_utils.save_model(new_model, MODEL_PATH)
            model = new_model  # Actualización del modelo después del reentrenamiento
            return jsonify({'message': 'Modelo reentrenado exitosamente', 'metrics': metrics})
        else:
            return jsonify({'error': 'Debe proporcionar un archivo válido para reentrenar'}), 400

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)