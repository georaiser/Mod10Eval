"""
Modulo para cargar un modelo entrenado y realizar predicciones.
"""

import joblib
import numpy as np
import os
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

class ModelPredictor:
    """
    Clase que maneja la carga del modelo, escalador y las predicciones.
    """
    
    def __init__(self, model_path="modelo.pkl", scaler_path="scaler.pkl", info_path="model_info.pkl"):
        """
        Inicializa el predictor cargando el modelo, escalador y su informacion.
        
        Args:
            model_path (str): Ruta del archivo del modelo
            scaler_path (str): Ruta del archivo del escalador
            info_path (str): Ruta del archivo con informacion del modelo
        """
        self.model = None
        self.scaler = None
        self.model_info = None
        self.load_model(model_path, scaler_path, info_path)
    
    def load_model(self, model_path, scaler_path, info_path):
        """
        Carga el modelo, escalador y su informacion desde archivos.
        
        Args:
            model_path (str): Ruta del archivo del modelo
            scaler_path (str): Ruta del archivo del escalador
            info_path (str): Ruta del archivo con informacion del modelo
        """
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"No se encontro el archivo del modelo: {model_path}")
            
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"No se encontro el archivo del escalador: {scaler_path}")
            
            if not os.path.exists(info_path):
                raise FileNotFoundError(f"No se encontro el archivo de informacion: {info_path}")
            
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.model_info = joblib.load(info_path)
            logging.info("Modelo, escalador y informacion cargados correctamente")
            logging.info(f"Informacion del modelo: {self.model_info}")
            
        except Exception as e:
            logging.error(f"Error al cargar el modelo: {str(e)}")
            raise
    
    def validate_input(self, features):
        """
        Valida que las caracteristicas de entrada sean correctas.
        
        Args:
            features (list): Lista de caracteristicas numericas
            
        Returns:
            bool: True si la entrada es valida, False en caso contrario
            str: Mensaje de error si la entrada no es valida
        """
        if not isinstance(features, list):
            return False, "Las caracteristicas deben ser una lista"
        
        if len(features) != self.model_info['n_features']:
            return False, f"Se esperan {self.model_info['n_features']} caracteristicas, se recibieron {len(features)}"
        
        try:
            # Verificar que todos los valores sean numericos
            [float(x) for x in features]
        except (ValueError, TypeError):
            return False, "Todas las caracteristicas deben ser valores numericos"
        
        return True, "Entrada valida"
    
    def predict(self, features):
        """
        Realiza una prediccion usando el modelo y escalador cargados.
        
        Args:
            features (list): Lista de caracteristicas numericas
            
        Returns:
            dict: Diccionario con la prediccion y confianza
        """
        # Convertir a array numpy
        features_array = np.array(features).reshape(1, -1)
        
        # Escalar las caracteristicas usando el escalador entrenado
        features_scaled = self.scaler.transform(features_array)
        
        # Realizar prediccion con datos escalados
        prediction = self.model.predict(features_scaled)[0]
        prediction_proba = self.model.predict_proba(features_scaled)[0]
        
        # Obtener nombre de la clase predicha
        class_name = self.model_info['target_names'][prediction]
        confidence = float(prediction_proba[prediction])
        
        return {
            'prediction': int(prediction),
            'class_name': class_name,
            'confidence': confidence
        }
