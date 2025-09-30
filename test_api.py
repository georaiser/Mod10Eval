"""
Script para probar la API REST de predicción de cáncer de mama.

Este script realiza pruebas automatizadas de la API usando diferentes
casos de prueba para validar su funcionamiento.
"""

import requests
import json
import time
from sklearn.datasets import load_breast_cancer


class APITester:
    """
    Clase para realizar pruebas automatizadas de la API.
    """
    
    def __init__(self, base_url="http://localhost:5000"):
        """
        Inicializa el tester con la URL base de la API.
        
        Args:
            base_url (str): URL base de la API
        """
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_connection(self):
        """
        Prueba la conexión básica con la API.
        
        Returns:
            bool: True si la conexión es exitosa
        """
        try:
            response = self.session.get(f"{self.base_url}/")
            if response.status_code == 200:
                print("✅ Conexión con la API exitosa")
                print(f"Respuesta: {response.json()}")
                return True
            else:
                print(f"❌ Error de conexión: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error al conectar con la API: {str(e)}")
            return False
    
    def test_model_info(self):
        """
        Prueba el endpoint de información del modelo.
        """
        try:
            response = self.session.get(f"{self.base_url}/info")
            if response.status_code == 200:
                print("✅ Endpoint /info funciona correctamente")
                info = response.json()
                print(f"Características del modelo: {info['n_features']}")
                print(f"Clases objetivo: {info['target_names']}")
            else:
                print(f"❌ Error en endpoint /info: {response.status_code}")
        except Exception as e:
            print(f"❌ Error al probar /info: {str(e)}")
    
    def test_prediction(self, features, description=""):
        """
        Prueba el endpoint de predicción con datos específicos.
        
        Args:
            features (list): Lista de características para predecir
            description (str): Descripción del caso de prueba
        
        Returns:
            dict: Respuesta de la API
        """
        try:
            data = {"features": features}
            response = self.session.post(
                f"{self.base_url}/predict",
                json=data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Predicción exitosa {description}")
                print(f"   Clase predicha: {result['class_name']}")
                print(f"   Confianza: {result['confidence']:.4f}")
                return result
            else:
                print(f"❌ Error en predicción {description}: {response.status_code}")
                print(f"   Respuesta: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error al probar predicción {description}: {str(e)}")
            return None
    
    def test_invalid_inputs(self):
        """
        Prueba la API con entradas inválidas para verificar el manejo de errores.
        """
        print("\n🔍 Probando manejo de errores...")
        
        # Test 1: JSON sin clave 'features'
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json={"data": [1, 2, 3]},
                headers={"Content-Type": "application/json"}
            )
            if response.status_code == 400:
                print("✅ Manejo correcto de JSON sin clave 'features'")
            else:
                print(f"❌ Error esperado no detectado: {response.status_code}")
        except Exception as e:
            print(f"❌ Error en test de JSON inválido: {str(e)}")
        
        # Test 2: Número incorrecto de características
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json={"features": [1, 2, 3]},  # Muy pocas características
                headers={"Content-Type": "application/json"}
            )
            if response.status_code == 400:
                print("✅ Manejo correcto de número incorrecto de características")
            else:
                print(f"❌ Error esperado no detectado: {response.status_code}")
        except Exception as e:
            print(f"❌ Error en test de características incorrectas: {str(e)}")
        
        # Test 3: Características no numéricas
        try:
            features = ["texto"] + [0.0] * 29  # Primera característica como texto
            response = self.session.post(
                f"{self.base_url}/predict",
                json={"features": features},
                headers={"Content-Type": "application/json"}
            )
            if response.status_code == 400:
                print("✅ Manejo correcto de características no numéricas")
            else:
                print(f"❌ Error esperado no detectado: {response.status_code}")
        except Exception as e:
            print(f"❌ Error en test de características no numéricas: {str(e)}")


def get_sample_data():
    """
    Obtiene datos de muestra del dataset para las pruebas.
    
    Returns:
        list: Lista de casos de prueba con sus características
    """
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    # Seleccionar algunos casos de prueba
    test_cases = [
        {
            "features": X[0].tolist(),  # Primer caso
            "expected": "maligno" if y[0] == 0 else "benigno",
            "description": "(Caso real #1)"
        },
        {
            "features": X[100].tolist(),  # Caso intermedio
            "expected": "maligno" if y[100] == 0 else "benigno",
            "description": "(Caso real #2)"
        },
        {
            "features": X[200].tolist(),  # Caso final
            "expected": "maligno" if y[200] == 0 else "benigno",
            "description": "(Caso real #3)"
        }
    ]
    
    return test_cases


def main():
    """
    Función principal que ejecuta todas las pruebas.
    """
    print("🧪 Iniciando pruebas de la API de predicción de cáncer de mama\n")
    
    # Inicializar tester
    tester = APITester()
    
    # Test 1: Probar conexión
    print("1. Probando conexión básica...")
    if not tester.test_connection():
        print("❌ No se puede conectar con la API. Asegúrate de que esté ejecutándose.")
        return
    
    print("\n" + "="*60)
    
    # Test 2: Probar información del modelo
    print("2. Probando endpoint de información...")
    tester.test_model_info()
    
    print("\n" + "="*60)
    
    # Test 3: Probar predicciones con datos reales
    print("3. Probando predicciones con datos reales...")
    test_cases = get_sample_data()
    
    for i, case in enumerate(test_cases, 1):
        print(f"\nCaso de prueba {i} {case['description']}:")
        result = tester.test_prediction(case['features'], case['description'])
        
        if result:
            print(f"   Esperado: {case['expected']}")
            print(f"   Obtenido: {result['class_name']}")
    
    print("\n" + "="*60)
    
    # Test 4: Probar manejo de errores
    print("4. Probando manejo de errores...")
    tester.test_invalid_inputs()
    
    print("\n" + "="*60)
    print("✅ Pruebas completadas")


if __name__ == "__main__":
    main()
