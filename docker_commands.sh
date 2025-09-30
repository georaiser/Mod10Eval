# Comandos para construir y ejecutar el contenedor Docker

# 1. Construir la imagen Docker
docker build -t breast-cancer-api .

# 2. Ejecutar el contenedor mapeando el puerto 5000
docker run -p 5001:5000 breast-cancer-api

# 3. Ejecutar en modo detached (en segundo plano)
docker run -d -p 5001:5000 --name cancer-api breast-cancer-api

# 4. Ver logs del contenedor
docker logs breast-cancer-api

# 5. Detener el contenedor
docker stop breast-cancer-api

# 6. Eliminar el contenedor
docker rm breast-cancer-api

# 7. Ver contenedores en ejecución
docker ps

# 8. Ver todas las imágenes
docker images

# 9. Eliminar la imagen
docker rmi breast-cancer-api