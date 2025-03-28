from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
import traceback
import model.model_utils as model_utils  # Funciones del modelo

app = Flask(__name__)

# Ruta de acceso relativa para el archivo modelo
MODEL_PATH = 'model/model.pkl'
ALLOWED_EXTENSIONS = {'xlsx', 'csv'}
app.config['UPLOAD_FOLDER'] = 'temp'
model = model_utils.load_model(MODEL_PATH)  # Cargar modelo al iniciar el servidor


# Función para verificar extensiones de archivos permitidos
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('indexapp.html')  # Página principal


@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.form.get('text')  # Texto para predicciones individuales
        opinion_file = request.files.get('opinion_file')  # Archivo de entrada
        predictions = {}

        if text:  # Entrada basada en texto
            data_instance = {'text_input': text}
            predictions = model_utils.predict(model, [data_instance])  # Predicción usando el modelo

        elif opinion_file and allowed_file(opinion_file.filename):  # Entrada basada en archivo
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(opinion_file.filename))
            opinion_file.save(file_path)  # Guardar archivo temporalmente
            predictions = model_utils.predict_from_file(model, file_path)  # Predicción usando archivo

        return jsonify({'predictions': predictions})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        labels_file = request.files.get('labels_file')  # Capturar archivo de datos etiquetados

        if labels_file and allowed_file(labels_file.filename):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(labels_file.filename))
            labels_file.save(file_path)  # Guardar archivo temporalmente
            new_model, metrics = model_utils.retrain_model(MODEL_PATH, file_path)  # Reentrenar modelo
            model_utils.save_model(new_model, MODEL_PATH)  # Guardar el modelo actualizado
            return jsonify({'message': 'Modelo reentrenado exitosamente', 'metrics': metrics})

        return jsonify({'error': 'Archivo no válido o no proporcionado'}), 400

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)