"""
API REST con Flask para consumir un modelo de clasificacion de cancer de mama.

Esta API recibe caracteristicas medicas y devuelve una prediccion sobre
si el tumor es benigno o maligno.
"""

from flask import Flask, request, jsonify

# Importar la clase ModelPredictor desde predict_model.py
from predict_model import ModelPredictor

# Crear instancia de Flask
app = Flask(__name__)

# Inicializar el predictor
try:
    predictor = ModelPredictor(model_path="modelo.pkl", scaler_path="scaler.pkl", info_path="model_info.pkl")
except Exception as e:
    print(f"Error al inicializar el predictor: {str(e)}")
    predictor = None


@app.route('/')
def home():
    """
    Endpoint principal que confirma que la API esta funcionando.
    
    Returns:
        dict: Mensaje de estado de la API
    """
    if predictor is None:
        return jsonify({
            'status': 'error',
            'message': 'El modelo no esta cargado correctamente'
        }), 500
    
    return jsonify({
        'status': 'success',
        'message': 'API de prediccion de cancer de mama lista',
        'endpoints': {
            'predict': '/predict (POST)',
            'info': '/info (GET)'
        }
    })


@app.route('/info')
def model_info():
    """
    Endpoint que devuelve informacion sobre el modelo.
    
    Returns:
        dict: Informacion del modelo cargado
    """
    if predictor is None or predictor.model_info is None:
        return jsonify({
            'error': 'Informacion del modelo no disponible'
        }), 500
    
    return jsonify({
        'model_type': 'RandomForestClassifier',
        'preprocessing': 'StandardScaler',
        'n_features': predictor.model_info['n_features'],
        'target_names': predictor.model_info['target_names'],
        'feature_names': predictor.model_info['feature_names']
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint principal para realizar predicciones.
    
    Recibe un JSON con caracteristicas y devuelve la prediccion.
    
    Returns:
        dict: Resultado de la prediccion en formato JSON
    """
    try:
        # Verificar que el predictor este disponible
        if predictor is None:
            return jsonify({
                'error': 'El modelo no esta disponible'
            }), 500
        
        # Verificar que la solicitud contenga JSON
        if not request.is_json:
            return jsonify({
                'error': 'La solicitud debe contener JSON valido'
            }), 400
        
        data = request.get_json()
        
        # Verificar que exista la clave 'features'
        if 'features' not in data:
            return jsonify({
                'error': 'Se requiere la clave "features" en el JSON'
            }), 400
        
        features = data['features']
        
        # Validar entrada
        is_valid, message = predictor.validate_input(features)
        if not is_valid:
            return jsonify({
                'error': message
            }), 400
        
        # Realizar prediccion
        result = predictor.predict(features)
        
        return jsonify({
            'status': 'success',
            'prediction': result['prediction'],
            'class_name': result['class_name'],
            'confidence': result['confidence']
        })
    
    except Exception as e:
        return jsonify({
            'error': f'Error interno del servidor: {str(e)}'
        }), 500


@app.errorhandler(404)
def not_found(error):
    """
    Manejador para errores 404.
    
    Returns:
        dict: Mensaje de error para rutas no encontradas
    """
    return jsonify({
        'error': 'Endpoint no encontrado'
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """
    Manejador para errores 405.
    
    Returns:
        dict: Mensaje de error para metodos no permitidos
    """
    return jsonify({
        'error': 'Metodo HTTP no permitido para este endpoint'
    }), 405


if __name__ == '__main__':
    #app.run(debug=True, host='0.0.0.0', port=5000)
    app.run(host='0.0.0.0', port=8080)
