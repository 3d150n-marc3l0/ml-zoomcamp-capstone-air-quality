# Usar una imagen base de Python
#FROM python:3.10.12-slim
FROM python:3.12-slim
#FROM python:3.12-alpine

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos del modelo, DictVectorizer y Pipfile
COPY ["Pipfile", "Pipfile.lock", "./"]

# Instalar pipenv
#RUN pip install --no-cache-dir pipenv

# Instalar las dependencias desde el Pipfile
#RUN pipenv install --deploy --ignore-pipfile
#RUN pipenv install --deploy --system

RUN pip install --no-cache-dir pipenv && \
    pipenv install --deploy --system

# Establecer la variable de entorno para Flask
#ENV FLASK_ENV=development
#ENV FLASK_APP=app.py 

# Copiar el código de la aplicación
#COPY app.py .
# Copia el código de la aplicación
COPY app.py .
COPY train.py .
COPY config/app_config.yaml config/app_config.yaml
COPY data/prepro/air-quality-madrid_total_transform.pkl data/prepro/air-quality-madrid_total_transform.pkl
COPY data/prepro/air-quality-madrid_total_train.csv data/prepro/air-quality-madrid_total_train.csv
COPY data/prepro/air-quality-madrid_total_valid.csv data/prepro/air-quality-madrid_total_valid.csv
COPY data/prepro/air-quality-madrid_total_test.csv  data/prepro/air-quality-madrid_total_test.csv
COPY models/tcn/baseline_model_tcn.pt models/tcn/baseline_model_tcn.pt
COPY models/tcn/best_model_tcn.pt models/tcn/best_model_tcn.pt
COPY models/nbeats/baseline_model_nbeats.pt models/nbeats/baseline_model_nbeats.pt
#RUN ls -lh && pwd

# Exponer el puerto en el que se ejecutará la aplicación
EXPOSE 5000

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Comando para ejecutar la aplicación
#CMD ["pipenv", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
#CMD ["tail", "-f", "/dev/null"]
CMD ["pipenv", "run", "python", "app.py"]
