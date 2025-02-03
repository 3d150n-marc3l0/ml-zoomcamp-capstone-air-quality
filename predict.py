import os
import yaml
import argparse
# Files
import json
import pickle 

import pandas as pd
import numpy as np

# Logging
import logging

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
log_file_path = os.path.join("logs",'predict-logs.log')
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



# Create models dict
MODEL_DICT = {
    "NBeats": NBEATSModel,
    "TCN": TCNModel
}

def run_predict(
        model,
        df_valid, 
        df_test,
        run_exp_name:str,
        output_dir:str,
        transform=None
):    
    # Predict
    # Evaluation Base Random Forest
    eval_results = run_evaluate_model(
        model,
        df_valid,
        df_test,
        transform=transform
    )
    
    # Save prediction
    # Save Metrics
    metrics_eval = eval_results['metrics']
    logger.info(f"[RUN-TRAINING] Test Metrics: {metrics_eval}")
    metrics_eval_path = os.path.join(output_dir, f'{run_exp_name}_eval_test.json')
    with open(metrics_eval_path, 'w') as f:
        json.dump(metrics_eval, f)

    # Save forecast
    forecast_eval = eval_results['forecast']
    forecast_eval_path = os.path.join(output_dir, f'{run_exp_name}_forecast_test.csv')
    forecast_eval.pd_dataframe().to_csv(forecast_eval_path)


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
    with open(args.config, 'r') as f:
        prediction_config = yaml.safe_load(f)

    # Results dir
    ouput_dir = prediction_config["results"]["prediction_dir"]
    os.makedirs(ouput_dir, exist_ok=True)

    # Datasets
    transfrom_path = prediction_config["data_config"][0]["transform_path"]
    train_ts_path = prediction_config["data_config"][0]["train_path"]
    valid_ts_path = prediction_config["data_config"][0]["valid_path"]
    test_ts_path  = prediction_config["data_config"][0]["test_path"]
    ts_name = prediction_config["data_config"][0]["series_name"]
    print(f"Time series name: {ts_name}")
    with open(transfrom_path, 'rb') as f:
        transform = pickle.load(f)
    train_ts = TimeSeries.from_csv(train_ts_path, time_col='ds')
    valid_ts = TimeSeries.from_csv(valid_ts_path, time_col='ds')
    test_ts  = TimeSeries.from_csv(test_ts_path, time_col='ds')

    # Trained Models
    TRAINING_MODELS_CONFIG = prediction_config["models"]
    
    # Iterate models
    for model_config in TRAINING_MODELS_CONFIG:
        # Read Setting
        logger.info(f"[PREDICTION] Config: {model_config}")
        model_name = model_config['model_name']
        logger.info(f'[PREDICTION] model_name: {model_name}')

        # Load model
        model_path = model_config['model_path']
        logger.info(f'[PREDICTION] model_path: {model_path}')
        # Instanciar el modelo usando getattr
        if model_name in MODEL_DICT:
            model = MODEL_DICT[model_name].load(model_path)
        else:
            raise ValueError(f"Model {model_name} not supported.")

        # Experiment name
        run_exp_name = os.path.splitext(os.path.basename(model_path))[0]
        run_exp_name = f'{run_exp_name}_{ts_name}'
        exp_results_path = os.path.join(ouput_dir, run_exp_name)
        os.makedirs(exp_results_path, exist_ok=True)

        # Run preductions
        run_predict(
            model,
            valid_ts, 
            test_ts,
            run_exp_name,
            output_dir=exp_results_path,
            transform=transform
        )


if __name__ == "__main__":
    main()

