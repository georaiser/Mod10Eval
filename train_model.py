"""
Script para entrenar un modelo de clasificacion de cancer de mama.

Este modulo se encarga de:
- Cargar el dataset de Breast Cancer de Wisconsin
- Preprocesar los datos utilizando StandardScaler
- Entrenar un modelo RandomForestClassifier
- Evaluar el rendimiento del modelo
- Guardar el modelo, escalador e informacion para posterior uso en la API
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configuracion del sistema de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DataLoader:
    """
    Clase responsable de cargar y preparar el dataset.
    
    Esta clase maneja la carga del dataset de cancer de mama de Wisconsin
    y proporciona metodos para acceder a los datos y metadatos.
    """
    
    def __init__(self):
        """Inicializa el cargador de datos."""
        self.data = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.target_names = None
        
    def load_dataset(self):
        """
        Carga el dataset de cancer de mama de Wisconsin.
        
        Returns:
            tuple: Datos de caracteristicas (X), etiquetas (y), nombres de caracteristicas, nombres de clases
        """
        try:
            logging.info("Cargando dataset de cancer de mama...")
            self.data = load_breast_cancer()
            self.X = self.data.data
            self.y = self.data.target
            self.feature_names = self.data.feature_names
            self.target_names = self.data.target_names
            
            logging.info(f"Dataset cargado exitosamente: {self.X.shape[0]} muestras, {self.X.shape[1]} caracteristicas")
            logging.info(f"Clases disponibles: {self.target_names}")
            
            return self.X, self.y, self.feature_names, self.target_names
            
        except Exception as e:
            logging.error(f"Error al cargar el dataset: {str(e)}")
            raise
    
    def get_dataset_info(self):
        """
        Obtiene informacion detallada del dataset.
        
        Returns:
            dict: Informacion del dataset incluyendo dimensiones y distribucion de clases
        """
        if self.X is None or self.y is None:
            raise ValueError("Dataset no cargado. Ejecute load_dataset() primero.")
            
        unique, counts = np.unique(self.y, return_counts=True)
        class_distribution = dict(zip(self.target_names[unique], counts))
        
        return {
            'n_samples': self.X.shape[0],
            'n_features': self.X.shape[1],
            'class_distribution': class_distribution,
            'feature_names': self.feature_names.tolist(),
            'target_names': self.target_names.tolist()
        }


class ModelTrainer:
    """
    Clase responsable del entrenamiento del modelo de Machine Learning.
    
    Esta clase maneja el preprocesamiento de datos, entrenamiento del modelo
    y evaluacion del rendimiento utilizando tecnicas de normalizacion.
    """
    
    def __init__(self, random_state=42):
        """
        Inicializa el entrenador de modelos.
        
        Args:
            random_state (int): Semilla para reproducibilidad de resultados
        """
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def prepare_data(self, X, y, test_size=0.2):
        """
        Prepara los datos dividiendolos en conjuntos de entrenamiento y prueba.
        
        Args:
            X (array): Caracteristicas del dataset
            y (array): Etiquetas del dataset
            test_size (float): Proporcion de datos para prueba (default: 0.2)
        """
        logging.info("Dividiendo datos en conjuntos de entrenamiento y prueba...")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state, 
            stratify=y
        )
        
        logging.info(f"Datos de entrenamiento: {self.X_train.shape[0]} muestras")
        logging.info(f"Datos de prueba: {self.X_test.shape[0]} muestras")
    
    def train_model(self, n_estimators=100, max_depth=10):
        """
        Entrena el modelo RandomForest con escalado de caracteristicas.
        
        Args:
            n_estimators (int): Numero de arboles en el bosque
            max_depth (int): Profundidad maxima de los arboles
            
        Returns:
            tuple: Modelo entrenado y escalador ajustado
        """
        if self.X_train is None:
            raise ValueError("Datos no preparados. Ejecute prepare_data() primero.")
            
        logging.info("Iniciando entrenamiento del modelo...")
        
        # Crear y ajustar el escalador con datos de entrenamiento
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        
        # Crear y entrenar el modelo con datos escalados
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            n_jobs=-1  # Utilizar todos los nucleos disponibles
        )
        
        self.model.fit(X_train_scaled, self.y_train)
        logging.info("Modelo entrenado exitosamente")
        
        return self.model, self.scaler
    
    def evaluate_model(self, target_names):
        """
        Evalua el rendimiento del modelo entrenado.
        
        Args:
            target_names (array): Nombres de las clases objetivo
            
        Returns:
            dict: Metricas de evaluacion del modelo
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Modelo no entrenado. Ejecute train_model() primero.")
            
        logging.info("Evaluando rendimiento del modelo...")
        
        # Escalar datos de prueba y realizar predicciones
        X_test_scaled = self.scaler.transform(self.X_test)
        y_pred = self.model.predict(X_test_scaled)
        
        # Calcular metricas
        accuracy = accuracy_score(self.y_test, y_pred)
        
        logging.info(f"Precision del modelo: {accuracy:.4f}")
        print(f"\nPrecision del modelo: {accuracy:.4f}")
        print("\nReporte de clasificacion:")
        print(classification_report(self.y_test, y_pred, target_names=target_names))
        
        # Crear y mostrar matriz de confusion
        self._plot_confusion_matrix(self.y_test, y_pred, target_names)
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'true_labels': self.y_test
        }
    
    def _plot_confusion_matrix(self, y_true, y_pred, target_names):
        """
        Crea y guarda la matriz de confusion.
        
        Args:
            y_true (array): Etiquetas verdaderas
            y_pred (array): Predicciones del modelo
            target_names (array): Nombres de las clases
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=target_names, yticklabels=target_names)
        plt.title('Matriz de Confusion - Modelo RandomForest')
        plt.ylabel('Etiqueta Real')
        plt.xlabel('Etiqueta Predicha')
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        logging.info("Matriz de confusion guardada como 'confusion_matrix.png'")


class ModelSaver:
    """
    Clase responsable de guardar el modelo entrenado y sus componentes.
    
    Esta clase maneja la persistencia del modelo, escalador e informacion
    necesaria para la API de prediccion.
    """
    
    @staticmethod
    def save_model_components(model, scaler, model_info, 
                            model_filename="modelo.pkl", 
                            scaler_filename="scaler.pkl",
                            info_filename="model_info.pkl"):
        """
        Guarda todos los componentes del modelo entrenado.
        
        Args:
            model: Modelo entrenado
            scaler: Escalador ajustado
            model_info (dict): Informacion del modelo
            model_filename (str): Nombre del archivo del modelo
            scaler_filename (str): Nombre del archivo del escalador
            info_filename (str): Nombre del archivo de informacion
        """
        try:
            # Guardar modelo
            joblib.dump(model, model_filename)
            logging.info(f"Modelo guardado como '{model_filename}'")
            
            # Guardar escalador
            joblib.dump(scaler, scaler_filename)
            logging.info(f"Escalador guardado como '{scaler_filename}'")
            
            # Guardar informacion del modelo
            joblib.dump(model_info, info_filename)
            logging.info(f"Informacion del modelo guardada como '{info_filename}'")
            
        except Exception as e:
            logging.error(f"Error al guardar componentes del modelo: {str(e)}")
            raise


def main():
    """
    Funcion principal que orquesta todo el proceso de entrenamiento.
    
    Esta funcion ejecuta el pipeline completo de entrenamiento:
    1. Carga de datos
    2. Preparacion de datos
    3. Entrenamiento del modelo
    4. Evaluacion del modelo
    5. Guardado de componentes
    """
    try:
        # Paso 1: Cargar datos
        data_loader = DataLoader()
        X, y, feature_names, target_names = data_loader.load_dataset()
        
        # Obtener informacion del dataset
        dataset_info = data_loader.get_dataset_info()
        print(f"Informacion del dataset:")
        print(f"- Muestras totales: {dataset_info['n_samples']}")
        print(f"- Caracteristicas: {dataset_info['n_features']}")
        print(f"- Distribucion de clases: {dataset_info['class_distribution']}")
        
        # Paso 2: Preparar y entrenar modelo
        trainer = ModelTrainer()
        trainer.prepare_data(X, y)
        model, scaler = trainer.train_model()
        
        # Paso 3: Evaluar modelo
        evaluation_results = trainer.evaluate_model(target_names)
        
        # Paso 4: Preparar informacion para guardar
        model_info = {
            'feature_names': feature_names.tolist(),
            'target_names': target_names.tolist(),
            'n_features': len(feature_names),
            'accuracy': evaluation_results['accuracy'],
            'model_type': 'RandomForestClassifier'
        }
        
        # Paso 5: Guardar componentes
        ModelSaver.save_model_components(model, scaler, model_info)
        
        print(f"\n Proceso de entrenamiento completado exitosamente")
        print(f"   Precision final: {evaluation_results['accuracy']:.4f}")
        
    except Exception as e:
        logging.error(f"Error en el proceso de entrenamiento: {str(e)}")
        print(f" Error en el entrenamiento: {str(e)}")
        raise


if __name__ == "__main__":
    main()
