from flask import Flask, request, jsonify
import os
import pickle
import json
import yaml
import numpy as np
import pandas as pd
# Logging
import logging
from darts import concatenate
from darts import TimeSeries
# Models
from darts.models import (
    TCNModel,
    TFTModel,
    NBEATSModel,
)
from train import run_evaluate_model

# Crear un logger
logger = logging.getLogger()

# Configuración para el archivo
os.makedirs("logs", exist_ok=True)
file_handler = logging.FileHandler(os.path.join("logs", 'app-logs.log'))
file_handler.setLevel(logging.INFO)  # Puedes ajustar el nivel aquí (INFO, DEBUG, etc.)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(file_formatter)

# Configuración para la consola
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # Igualmente, puedes ajustar el nivel de consola
console_handler.setFormatter(file_formatter)

# Agregar ambos handlers al logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Establecer el nivel de log para el logger
logger.setLevel(logging.INFO)  # Este es el nivel general del logger (puedes ajustarlo)


# Inicializar la aplicación Flask
app = Flask(__name__)


# Create models dict
ALL_MODELS = {
    "NBeats": NBEATSModel,
    "TCN": TCNModel
}


def make_setting(config_path):
    # Read YAML file
    with open(config_path, 'r') as f:
        APP_CONFIG = yaml.safe_load(f)

    # Read the JSON file
    logger.info(f"app_config: {APP_CONFIG}")

    # Setting features
    # Datasets
    transfrom_path = APP_CONFIG["data_config"][0]["transform_path"]
    train_ts_path = APP_CONFIG["data_config"][0]["train_path"]
    valid_ts_path = APP_CONFIG["data_config"][0]["valid_path"]
    test_ts_path  = APP_CONFIG["data_config"][0]["test_path"]
    # Read transformer
    # print(f"[APP] Time series name: {ts_name}")
    global TRANSFORM   # Transform
    global TS          # Time series
    with open(transfrom_path, 'rb') as f:
        TRANSFORM = pickle.load(f)
    # Read dataset
    train_ts = TimeSeries.from_csv(train_ts_path, time_col='ds')
    valid_ts = TimeSeries.from_csv(valid_ts_path, time_col='ds')
    TS = concatenate([train_ts, valid_ts], axis=0)

    # Load models
    global AVAILABLE_MODELS
    AVAILABLE_MODELS = {}
    models = APP_CONFIG["models"]
    for model_conf in models:
        model_name = model_conf["model_name"]
        model_path = model_conf["model_path"]
        if not os.path.exists(model_path):
            logger.info(f"File {model_path} doesn't exist")
            continue
        # Load model
        if model_name in ALL_MODELS:
            model = ALL_MODELS[model_name].load(model_path)
        else:
            raise ValueError(f"Model {model_name} not supported.")
        # Save model
        AVAILABLE_MODELS[model_name] = model

CONFIG_PATH = os.path.join("config", "app_config.yaml")
make_setting(CONFIG_PATH)


# Define prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON from the request body
    data = request.get_json()
    logger.info(f"request: {data}")

    # Check that the 'days' parameter is present in the JSON
    if 'days' not in data:
        return jsonify({'error': 'The "days" parameter is required.'}), 400

    try:
        # Number of days to predict
        days_to_predict = int(data['days'])
        logger.info(f"[PREDICT] days: {days_to_predict}")

        # Check that the number of days is positive
        if days_to_predict <= 0:
            return jsonify({'error': 'The number of days must be a positive integer.'}), 400

        # Realizar la predicción con ambos modelos
        predictions = {}
        for model_name, model in AVAILABLE_MODELS.items():
            # Make the prediction
            forecast = model.predict(n=days_to_predict, series=TS)

            # Transform Back
            forecast = TRANSFORM.inverse_transform(forecast)

            # Convert the forecast to a DataFrame with 'ds' index and 'y' column
            forecast_df = forecast.pd_dataframe().reset_index()

            # Format the 'ds' column to your desired date format
            # For example, format to 'YYYY-MM-DD'
            forecast_df['ds'] = forecast_df['ds'].dt.strftime('%Y-%m-%d')

            # Convert the DataFrame to JSON, ensuring the data is serializable
            forecast_json = forecast_df.to_dict(orient='records')

            # Convert the DataFrame to JSON, ensuring the data is serializable
            predictions[model_name] = forecast_json
        # Return the prediction as JSON
        return jsonify({
            'predictions': predictions
        })

    except ValueError:
        return jsonify({'error': 'Invalid value for "days". It must be an integer.'}), 400



# Correr la aplicación
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
