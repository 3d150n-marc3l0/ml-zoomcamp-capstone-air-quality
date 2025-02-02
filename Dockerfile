# Usar una imagen base de Python
FROM python:3.10.12-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos del modelo, DictVectorizer y Pipfile
COPY ["Pipfile", "Pipfile.lock", "./"]

# Instalar pipenv
RUN pip install --no-cache-dir pipenv

# Instalar las dependencias desde el Pipfile
RUN pipenv install --deploy --ignore-pipfile
#RUN pipenv install --deploy --system

# Establecer la variable de entorno para Flask
#ENV FLASK_ENV=development
#ENV FLASK_APP=app.py 

# Copiar el código de la aplicación
#COPY app.py .
# Copia el código de la aplicación
COPY app.py .
COPY config/ config/
COPY models/ models/
#RUN ls -lh && pwd

# Exponer el puerto en el que se ejecutará la aplicación
EXPOSE 5000

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Comando para ejecutar la aplicación
#CMD ["pipenv", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
#CMD ["tail", "-f", "/dev/null"]
CMD ["pipenv", "run", "python", "app.py"]