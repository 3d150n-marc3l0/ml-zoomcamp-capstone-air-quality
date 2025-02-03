import os
import yaml
import argparse
import csv
import json
import pandas as pd
import requests
from darts import TimeSeries

# Logging
import logging

# Crear un logger
logger = logging.getLogger()

# Configuración para el archivo
os.makedirs("logs", exist_ok=True)
log_file_path = os.path.join("logs",'app-test-logs.log')
file_handler = logging.FileHandler(log_file_path)
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

def send_json_to_flask(json_data, url, output_dir):
    """Send JSON a una API Flask usando una solicitud POST."""
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, json=json_data, headers=headers)
    
    if response.status_code == 200:
        logger.info("Successful response.")
        test_res_path = os.path.join(output_dir, "test_app_ok.json")
        logger.info(f"Saving successful response to {test_res_path}")
        with open(test_res_path, 'w') as f:
            json.dump(response.json(), f)
    else:
        logger.error(f"Error response wiyh code {response.status_code}")
        error_res_path = os.path.join(output_dir, "test_app_error.json")
        logger.error(f"Saving error response to {error_res_path}")
        with open(error_res_path, "") as f:
            f.write(response.text)



def main():
    # Create parser
    parser = argparse.ArgumentParser(description="Programa que recibe tres parámetros: fichero de datos, tipo de algoritmo y directorio de salida.")
    
    # Definir los argumentos obligatorios
    parser.add_argument('config', type=str, help="El fichero de datos (raw_data) que se va a procesar")

    # Parsear los argumentos
    args = parser.parse_args()

    # Read raw data
    # Verificar si el fichero de datos existe
    if not os.path.isfile(args.config):
        print(f"Error: File {args.config} doesn't exist.")
        return

    # Read YAML file
    logger.info(f"Reading file {args.config}")
    with open(args.config, 'r') as f:
        app_test_config = yaml.safe_load(f)

    OUTPUT_DIR = app_test_config["outputs"]["path"]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Datasets
    transfrom_path = app_test_config["data_config"][0]["transform_path"]
    train_ts_path = app_test_config["data_config"][0]["train_path"]
    valid_ts_path = app_test_config["data_config"][0]["valid_path"]
    test_ts_path  = app_test_config["data_config"][0]["test_path"]
    # Read dataset
    train_ts = TimeSeries.from_csv(train_ts_path, time_col='ds')
    valid_ts = TimeSeries.from_csv(valid_ts_path, time_col='ds')
    test_ts  = TimeSeries.from_csv(test_ts_path, time_col='ds')

    days_to_predict = len(test_ts)

    json_data = {
        "days": days_to_predict
    }

    endpoint = app_test_config["endpoints"]["predict"]

    logger.info(f"Sending data to: {endpoint}")
    # Enviar el JSON a la API Flask
    send_json_to_flask(json_data, endpoint, OUTPUT_DIR)

if __name__ == "__main__":
    main()
    
