# Usar imagen base oficial de Python
FROM python:3.9-slim

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar el archivo de requerimientos
COPY requirements.txt .

# Instalar las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todos los archivos del proyecto al contenedor
COPY . .

# Exponer el puerto 5000 para la aplicación Flask
#EXPOSE 5000
EXPOSE 8080

# Comando para ejecutar la aplicación
CMD ["python", "app.py"]