# Files
import glob
import pickle
import json
import shutil
import os
import yaml

# Console
import argparse

# Utils
import random
import time
from datetime import datetime
from itertools import product

import warnings

# Padans
import pandas as pd
import numpy as np

# Track
from tqdm.auto import tqdm

# Typing
from typing import Any, Dict, Optional, Callable

# Logging
import logging

# Plot
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

# Darts

from darts import TimeSeries
from darts.utils.missing_values import fill_missing_values
from darts.utils.callbacks import TFMProgressBar
from darts.utils.losses import SmapeLoss
from darts import concatenate
# Utils
from darts.utils.timeseries_generation import (
    datetime_attribute_timeseries,
    sine_timeseries,
)
# Metrics
from darts.metrics import (
    smape,
    mape,
    rmse,
    mae,
    r2_score
) 


# Models
from darts.models import (
    ARIMA,
    FFT,
    TCNModel,
    TFTModel,
    ExponentialSmoothing,
    KalmanForecaster,
    LightGBMModel,
    LinearRegressionModel,
    NaiveSeasonal,
    NBEATSModel,
    RandomForest,
    Theta,
)

# Transform
from darts.dataprocessing.transformers import (
    InvertibleMapper,
    Mapper,
    MissingValuesFiller,
    Scaler,
)

# Statmodels
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing

# Tensorflows
#import tensorflow as tf
#import tensorboard

# Pytorch
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# Optimization
import optuna
#from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
#from hyperopt.pyll import scope

# Crear un logger
logger = logging.getLogger()

# Configuración para el archivo
os.makedirs("logs", exist_ok=True)
log_file_path = os.path.join("logs",'train-logs.log')
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


def get_caller(function_name: str) -> Optional[Callable]:
    """
    Retrieves a function object by its name, checking first in the local scope 
    and then in the global scope.

    :param function_name: The name of the function to retrieve.
    :return: The function object if found and callable, otherwise None.
    """
    local_vars = locals()
    func = local_vars.get(function_name) or globals().get(function_name)

    return func if callable(func) else None


def read_model_parms(path: str) -> Dict[str, Any]:
    """
    Reads a JSON file and returns its content as a dictionary.
    
    :param path: Path to the JSON file.
    :return: Dictionary with the JSON content.
    """
    try:
        with open(path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ValueError(f"Error reading JSON: {e}")


def build_model_params(
    optimized_params_path: str, 
    default_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Reads a JSON file and merges its content with a dictionary of default values.
    If there are duplicate keys, the values from the JSON take precedence.
    
    :param optimized_params_path: Path to the JSON file.
    :param default_values: Dictionary with default values (optional).
    :return: Dictionary with the merged values.
    """
    optimized_params = read_model_parms(optimized_params_path)
    
    if default_params is None:
        return optimized_params
    
    # Merge dictionaries, giving priority to the JSON
    combined_params = {**default_params, **optimized_params}
    return combined_params


def read_air_quality_dataset(data_dir: str):
    """
    Reads all CSV files in a specified directory and concatenates them into a single DataFrame.

    This function searches for `.csv` files within the provided directory,
    reads them, and combines them into a single pandas DataFrame. It is useful for 
    merging multiple related data files, such as air quality data.

    Parameters:
    data_dir (str): The directory path where the CSV files are located.

    Returns:
    pd.DataFrame: A DataFrame containing the combined data from all the read CSV files.

    Example:
    >>> df = read_air_quality_data("/path/to/data")
    >>> df.head()

    Exceptions:
    - If no CSV files are found in the directory, the returned DataFrame will be empty.
    - If an error occurs while reading the CSV files, a pandas exception will be raised.


    Requires:
    - pandas (pd)
    - glob
    """
    # Path pattern for the CSV files
    csv_files = glob.glob(f'{data_dir}/*.csv')

    # Read all files and concatenate them into a single DataFrame
    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    return df


def check_stationarity(series, alpha=0.05):
    """
    Performs ADF and KPSS tests on a time series and determines if it is stationary.

    Parameters:
        - series: Time series (Pandas Series or list of values).
        - alpha: Significance level (default 0.05).

    Returns:
        - True if the series is stationary, False if it is not.
    """

    # ADF test (Augmented Dickey-Fuller)
    adf_stat, adf_pvalue, _, _, adf_critical_values, _ = adfuller(series, autolag='AIC')

    # Suppress warnings (to avoid InterpolationWarning from KPSS)
    warnings.filterwarnings("ignore", category=UserWarning)

    # KPSS test with a larger number of lags (adjustable)
    kpss_stat, kpss_pvalue, _, kpss_critical_values = kpss(series, regression='c', nlags=20)  # You can increase this number if needed

    # Stationarity evaluation
    adf_stationary = adf_pvalue < alpha  # We want this to be True (reject H0: non-stationary)
    kpss_stationary = kpss_pvalue > alpha  # We want this to be True (do not reject H0: stationary)

    # The series is stationary only if both tests indicate stationarity
    return adf_stationary and kpss_stationary


def detect_outliers_zscore(x, window_size=24, threshold=3):
    """
    Detects outliers in a time series using the Z-score method.

    This function applies a rolling window over the input time series `x`, calculates the 
    mean and standard deviation of the windowed data, then computes the Z-scores for each 
    data point. Anomalies are identified when the absolute value of the Z-score exceeds 
    a specified threshold.

    Parameters:
    x (pd.Series): The input time series data (usually a pandas Series).
    window_size (int, optional): The size of the rolling window to calculate the mean and 
                                  standard deviation (default is 24).
    threshold (float, optional): The Z-score threshold above which a data point is considered 
                                  an outlier (default is 3).

    Returns:
    pd.Series: A boolean Series where `True` indicates an anomaly (outlier) and `False` indicates normal data.

    Example:
    >>> anomalies = detect_outliers_zscore(time_series_data, window_size=12, threshold=3)
    >>> anomalies.head()

    Exceptions:
    - If `x` is not a pandas Series or does not contain numeric data, an error may be raised.

    Requires:
    - pandas (pd)
    - numpy (np)
    """
    # Apply rolling window to calculate the mean and standard deviation
    r = x.rolling(window=window_size)
    m = r.mean().shift(1)  # Mean of the rolling window
    s = r.std(ddof=0).shift(1)  # Standard deviation of the rolling window

    # Calculate the Z-scores
    zscores = (x - m) / s

    # Apply a threshold for the Z-score to identify anomalies
    anomalies = np.abs(zscores) > threshold

    return anomalies


def find_stationary_features(df: pd.Series, alpha = 0.05) -> pd.Series:
    print(f"Series: {df.shape} {df.columns}")
    no_stationary_cols = []
    stationary_cols = []
    for col in df.columns:
       # Test Stattionary
       series = df[[col]]
       is_stationary = check_stationarity(series, alpha=alpha)
       print(f"[{col}] is_stationary: {is_stationary}")
       if is_stationary:
         stationary_cols.append(col)
       else:
         no_stationary_cols.append(col)
    return no_stationary_cols, stationary_cols


def run_data_wrangling(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses a DataFrame by setting the 'date' column as the index,
    filtering out unnecessary columns, and imputing missing values.

    This function performs the following operations on the input DataFrame `df`:
    1. Converts the 'date' column to datetime format and sets it as the index.
    2. Filters the DataFrame to include only a specific set of columns related to air quality.
    3. Interpolates missing values in the selected columns using linear interpolation.

    Parameters:
    df (pd.DataFrame): The input DataFrame that contains the air quality data with a 'date' column.

    Returns:
    pd.DataFrame: A cleaned DataFrame with the 'date' column as the index, only the relevant columns,
                  and missing values interpolated.

    Example:
    >>> clean_df = preprocess(raw_df)
    >>> clean_df.head()

    Exceptions:
    - If the 'date' column is missing or improperly formatted, the function will raise an error.
    - If there are columns not present in the `filtered_columns` list, they will be ignored.

    Requires:
    - pandas (pd)
    """
    # Set 'date' as the index
    raw_df['date'] = pd.to_datetime(raw_df.date, infer_datetime_format=True)
    raw_df.set_index('date', inplace=True)
    raw_df.index.names = ['ds']

    # Select only the relevant columns related to air quality measurements
    # We filter the columns that have many null values ​​and it is not possible to perform any type of imputation.
    filtered_columns = ['EBE', 'TOL', 'BEN', 'NMHC', 'TCH', 'CO', 'SO_2', 'PM10', 'O_3', 'NO_2']
    clean_df = raw_df[filtered_columns]

    # Impute missing values using linear interpolation
    clean_df = clean_df.interpolate(method="linear")

    # Calculate total pollution
    clean_feats_daily_df = clean_df.resample('D').mean()


    # Test Stattionality
    alpha = 0.05
    no_stationary_cols = []
    stationary_cols = filtered_columns
    '''
    ori_no_stationary_cols, ori_stationary_cols = find_stationary_features(
        clean_feats_daily_df,
        alpha = alpha
    )
    logger.info(f"[ORI] No Stationary: {ori_no_stationary_cols}")
    logger.info(f"[ORI] stationary   : {ori_stationary_cols}")

    if len(ori_no_stationary_cols) > 0:
      # Make diffrence
      diff_cols = []
      for col in ori_no_stationary_cols:
          diff_col = f'{col}_diff'
          clean_feats_daily_df[diff_col] = clean_feats_daily_df[[col]].diff()
          diff_cols.append(diff_col)
      # Test Stattionality
      diff_no_stationary_cols, diff_stationary_cols = find_stationary_features(
          clean_feats_daily_df[diff_cols][1:],
          alpha = alpha
      )
      logger.info(f"[DIFF] No stationary: {diff_no_stationary_cols}")
      logger.info(f"[DIFF] stationary   : {diff_stationary_cols}")

    no_stationary_cols = ori_no_stationary_cols + diff_no_stationary_cols
    stationary_cols = ori_stationary_cols + diff_stationary_cols
    logger.info(f"No Stationary: {no_stationary_cols}")
    logger.info(f"Stationary   : {stationary_cols}")
    '''
    # Create total
    clean_feats_daily_df['total'] = clean_feats_daily_df[stationary_cols].sum(axis=1)
    if not check_stationarity(clean_feats_daily_df['total'], alpha=alpha):
      logger.info(f"[Total] No Stationary")
    else:
      #stationary_cols.append('total')
      logger.info(f"Total] Stationary")
    stationary_cols.append('total')
    # Change names colums and index
    #df.columns = ['y']
    #df.index.names = ['ds']



    # Remove NA
    clean_feats_daily_df = clean_feats_daily_df.dropna()

    # Test stationary
    # Remove outliers
    for col in stationary_cols:
        outliers = detect_outliers_zscore(clean_feats_daily_df[col], window_size=24)
        logger.info(f"[{col}] Outliers: {outliers.sum()}")
        # We replace the outliers in the series by linear interpolation
        clean_feats_daily_df[[col]].loc[outliers] = np.nan
        clean_feats_daily_df[[col]] = clean_feats_daily_df[[col]].interpolate(method='linear')

    # Return the cleaned DataFrame
    return {
        'stationary_cols': stationary_cols,
        'no_stationary_cols': no_stationary_cols,
        'series': clean_feats_daily_df
    }



def run_partition_data(df, train_ratio=0.6, val_ratio=0.2, test_ratio=None):
    """
    Splits a given dataset of multiple time series into training, validation, and test sets based on the specified ratios.
    The data is transformed into time series format using the `TimeSeries.from_dataframe` method.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame where the index is the time ('ds') and each column represents a different time series.
        Each column corresponds to a specific time series (e.g., 'EBE', 'TOL', 'BEN', etc.).

    train_ratio : float, optional, default=0.6
        The proportion of the dataset to be used for the training set. The value must be between 0 and 1.

    val_ratio : float, optional, default=0.2
        The proportion of the dataset to be used for the validation set. The value must be between 0 and 1.

    test_ratio : float, optional, default=None
        The proportion of the dataset to be used for the test set. If None, the remaining data after
        splitting into train and validation sets will be used for testing.

    Returns:
    --------
    dict : dictionary
        A dictionary where the keys are the time series identifiers (e.g., 'EBE', 'TOL', etc.) and the values
        are tuples containing TimeSeries objects for the training, validation, and test sets for each time series.

    Notes:
    ------
    - The dataset must have a datetime index (`ds`) and each column represents a time series.
    """

    # Check if DataFrame is empty
    if df.empty:
        raise ValueError("The input dataframe is empty!")

    # Validate that the dataframe contains a datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be of type DatetimeIndex.")

    # Normalize ratios if no test_ratio is given
    if test_ratio is None:
        test_ratio = 1 - (train_ratio + val_ratio)

    # Check if the sum of the ratios is less than or equal to 1
    if train_ratio + val_ratio + test_ratio > 1:
        raise ValueError("The sum of train_ratio, val_ratio, and test_ratio cannot exceed 1.")

    # Create a dictionary to store the TimeSeries splits for each series
    time_series_dict = {}

    # Get the number of rows in the dataset (this corresponds to the length of each time series)
    dataset_length = len(df)

    # Calculate split indices
    train_size = int(dataset_length * train_ratio)
    valid_size = int(dataset_length * val_ratio)
    test_size = dataset_length - (train_size + valid_size)

    index_name = 'ds'
    target_name = 'y'
    # Iterate over each column (time series) in the dataframe
    for feature in df.columns:
        # Split the data for the current time series
        series_data = df[[feature]]
        series_data.rename(columns={feature: target_name}, inplace=True)
        series_data = series_data.reset_index()

        # Split the series into train, validation, and test sets
        train = series_data[:train_size]
        valid = series_data[train_size:train_size + valid_size]
        test = series_data[train_size + valid_size:]

        # Convert pandas Series into TimeSeries objects for each split
        train_ts = TimeSeries.from_dataframe(train, time_col=index_name, value_cols=target_name)
        valid_ts = TimeSeries.from_dataframe(valid, time_col=index_name, value_cols=target_name)
        test_ts = TimeSeries.from_dataframe(test, time_col=index_name, value_cols=target_name)

        # Store the TimeSeries splits in the dictionary
        time_series_dict[feature] = (train_ts, valid_ts, test_ts)

    return time_series_dict


def run_transformation(ts_dict: pd.DataFrame):
    # Tranformation
    ts_scaled_dict = {}
    for feature, (train_ts, valid_ts, test_ts) in ts_dict.items():
       transform = Scaler()
       train_ts_scaled = transform.fit_transform(train_ts)
       valid_ts_scaled = transform.transform(valid_ts)
       test_ts_scaled  = transform.transform(test_ts)
       ts_scaled_dict[feature] = (transform, train_ts_scaled, valid_ts_scaled, test_ts_scaled)

    return ts_scaled_dict


def run_evaluate_model(
    model,
    series_ts: TimeSeries,
    test_ts: TimeSeries,
    horizont: int = 7,
    stride: int = 7,
    transform: Optional[Scaler] = None,
):
    # concatenate cal and test set to be able to start forecasting at the `test` start time
    all_series_ts = concatenate([series_ts, test_ts], axis=0)

    # Make predictions
    forecast = model.historical_forecasts(
        series=all_series_ts,
        forecast_horizon=horizont,
        stride=stride,
        start=test_ts.start_time(),
        #last_points_only=True,  # returns a single TimeSeries
        last_points_only=False,
        retrain=False,
        verbose=True,
    )
    forecast = concatenate(forecast)

    # Metrics
    val_mae = mae(test_ts, forecast)
    val_rmse = rmse(test_ts, forecast)
    val_mape = mape(test_ts, forecast)
    #val_r2_score = r2_score(test_ts.univariate_component(0), forecast)

    # Transform
    forecast_back = None
    if transform:
        logger.info('transform forecast')
        forecast_back = transform.inverse_transform(forecast)
        #forecast = transform.inverse_transform(forecast)
        #test_ts = transform.inverse_transform(test_ts)

    # Return    
    return {
        'forecast': forecast,
        'forecast_back': forecast_back,
        'metrics': {
          'mae': val_mae,
          'rmse': val_rmse,
          'mape': val_mape,
          #'r2_score': val_r2_score,
        }
    }


def run_traininig(
    df_train,
    df_valid,
    df_test,
    trainer,
    model_params: dict,
    model_name: str,
    models_dir:str,
    transform = None,
):
    # Train model
    model_params['model_name'] = model_name
    trained_result = trainer(
        df_train,
        df_valid,
        model_params,
        transform,
    )

    # Save Metrics
    metrics_train = trained_result['metrics']
    logger.info(f"[RUN-TRAINING] Train Metrics: {metrics_train}")
    metrics_eval_path = os.path.join(models_dir, f'{model_name}_eval_valid.json')
    with open(metrics_eval_path, 'w') as f:
        json.dump(metrics_train, f)

    # Save Model
    model = trained_result['model']
    model_path = os.path.join(models_dir, f'{model_name}.pt')
    model.save(model_path)

    # Save forecast
    forecast = trained_result['forecast']
    forecast_path = os.path.join(models_dir, f'{model_name}_forecast_valid.csv')
    forecast.pd_dataframe().to_csv(forecast_path)
    forecast_back = trained_result['forecast_back']
    forecast_back_path = os.path.join(models_dir, f'{model_name}_forecast_valid_back.csv')
    forecast_back.pd_dataframe().to_csv(forecast_back_path)

    # Evaluation Base Random Forest
    eval_results = run_evaluate_model(
        model,
        df_valid,
        df_test,
        transform=transform
    )

    # Save Metrics
    metrics_eval = eval_results['metrics']
    logger.info(f"[RUN-TRAINING] Test Metrics: {metrics_eval}")
    metrics_eval_path = os.path.join(models_dir, f'{model_name}_eval_test.json')
    with open(metrics_eval_path, 'w') as f:
        json.dump(metrics_eval, f)

    # Save forecast
    forecast_eval = eval_results['forecast']
    forecast_eval_path = os.path.join(models_dir, f'{model_name}_forecast_test.csv')
    forecast_eval.pd_dataframe().to_csv(forecast_eval_path)
    forecast_eval_back = eval_results['forecast_back']
    forecast_eval_back_path = os.path.join(models_dir, f'{model_name}_forecast_test_back.csv')
    forecast_eval_back.pd_dataframe().to_csv(forecast_eval_back_path)


def run_optimization(
    train_ts,
    valid_ts,
    optimizer,
    default_params,
    model_name,
    models_dir,
    transform,
    n_trials=50,
):
    # Optimize
    optimize_results = optimizer(
        train_ts,
        valid_ts,
        transform=transform,
        experiment_name=model_name,
        n_trials=n_trials,
        default_params=default_params
    )
    # Retrive best parametres
    best_opt_params = optimize_results['best_params']
    logger.info(f"[OPIMIZATION] Best params: {best_opt_params}")

    best_params = {**default_params, **best_opt_params}
    logger.info(f"[OPIMIZATION] Best params: {best_params}")

    # Save best parameters
    best_params_file = os.path.join(models_dir, f'{model_name}_optimized_params.json')
    with open(best_params_file, 'w') as f:
        json.dump(best_params, f)



def train_tcn(
    train,
    valid,
    params:dict={},
    transform=None
):
    # Preprocessing
    logger.info(f'params: {params}')

    # reproducibility
    torch.manual_seed(42)

    # some fixed parameters that will be the same for all models
    MAX_SAMPLES_PER_TS = 60

    # throughout training we'll monitor the validation loss for early stopping
    early_stopper = EarlyStopping(
        "val_loss",
        min_delta=0.001, patience=5,
        verbose=True
    )
    if 'callbacks' in params:
      callbacks = [early_stopper] + params['callbacks']
      del params['callbacks']
    else:
      callbacks = [early_stopper]

    # detect if a GPU is available
    if torch.cuda.is_available():
        pl_trainer_kwargs = {
            "accelerator": "gpu",
            "gpus": -1,
            "auto_select_gpus": True,
            "callbacks": callbacks,
        }
        num_workers = 4
    else:
        pl_trainer_kwargs = {"callbacks": callbacks}
        num_workers = 0

    # Check learning rate
    if 'learning_rate' in params:
      params['optimizer_kwargs'] = {'lr': params['learning_rate']}
      del params['learning_rate']

    # optionally also add the day of the week (cyclically encoded) as a past covariate
    #encoders = {"cyclic": {"past": ["dayofweek"]}} if include_dayofweek else None
    if 'include_dayofweek' in params:
      encoders = {"cyclic": {"past": ["dayofweek"]}}
      params['add_encoders'] = encoders
      del params['include_dayofweek']


    # build the TCN model
    model = TCNModel(
        **params
    )

    # train the model
    model.fit(
        series=train,
        val_series=valid,
        max_samples_per_ts=MAX_SAMPLES_PER_TS,
        dataloader_kwargs={"num_workers": num_workers},
    )

    # Reload best model over course of training
    model = TCNModel.load_from_checkpoint(params['model_name'])

    # Predict
    eval_valid = run_evaluate_model(
        model,
        train,
        valid,
        horizont=7,
        stride=7,
        transform=transform
    )
    
    train_result = {'model': model}
    return {**train_result, **eval_valid}


# Optimization with Optuna
def optimize_tcn_with_optuna(
    train_ts,
    valid_ts,
    transform,
    experiment_name='tcn_model',
    n_trials=50,
    #n_epochs=100,
    #random_seed=42
    default_params:dict=None
):
    # Función objetivo para Optuna
    def objective(trial):
        input_chunk_length = trial.suggest_int('input_chunk_length', 24, 60)
        output_chunk_length = trial.suggest_int('output_chunk_length', 12, 24)
        num_filters = trial.suggest_int('num_filters', 32, 256)
        kernel_size = trial.suggest_int('kernel_size', 2, 8)
        dropout = trial.suggest_uniform('dropout', 0.0, 0.5)
        batch_size = trial.suggest_int('batch_size', 16, 128)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)

        model_name = f"{experiment_name}_trial_{trial.number}"
        trial_params = {
            'input_chunk_length': input_chunk_length,
            'output_chunk_length': output_chunk_length,
            'num_filters': num_filters,
            'kernel_size': kernel_size,
            'dropout': dropout,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            #'optimizer_kwargs': {'lr': learning_rate},
            #'n_epochs': n_epochs,
            #'random_state': random_seed,
            # Model Name
            'model_name': model_name,
            'force_reset':True,
            'save_checkpoints':True,
        }
        if default_params:
          trial_params = {**default_params, **trial_params}
        
        # Evaluación con historical_forecasts
        #model = train_nbeats_model(series_scaled, input_chunk_length, output_chunk_length, num_blocks, thetas_dim, hidden_layer_units, dropout, learning_rate)
        results = train_tcn(
            train_ts,
            valid_ts,
            trial_params,
            transform
        )
        scores = results['metrics']
        error = scores['rmse']

        return error

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial), n_trials=n_trials)

    return {
        'best_params': study.best_params,
        'results': study.trials
    }


    
def train_nbeats(
    train,
    valid,
    params:dict={},
    transform=None
):
    logger.info(f"params: {params}")
    # reproducibility
    torch.manual_seed(42)

    # throughout training we'll monitor the validation loss for early stopping
    early_stopper = EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,
        patience=5,
        verbose=True
    )
    if 'callbacks' in params:
      callbacks = [early_stopper] + params['callbacks']
      del params['callbacks']
    else:
      callbacks = [early_stopper]

    # detect if a GPU is available
    if torch.cuda.is_available():
        pl_trainer_kwargs = {
            "accelerator": "gpu",
            "gpus": -1,
            "auto_select_gpus": True,
            "callbacks": callbacks,
        }
        num_workers = 4
    else:
        pl_trainer_kwargs = {"callbacks": callbacks}
        num_workers = 0

    params['pl_trainer_kwargs'] = pl_trainer_kwargs

    # Check learning rate
    if 'learning_rate' in params:
      params['optimizer_kwargs'] = {'lr': params['learning_rate']}
      del params['learning_rate']

    # Create model
    model = NBEATSModel(**params)

    # Train
    model.fit(
        train,
        val_series=valid
    )

    # reload best model over course of training
    model = NBEATSModel.load_from_checkpoint(params['model_name'])

    # Predict
    eval_valid = run_evaluate_model(
        model,
        train,
        valid,
        horizont=7,
        stride=7,
        transform=transform
    )

    train_result = {'model': model}
    return {**train_result, **eval_valid}


# Optimization with Optuna
def optimize_nbeats_with_optuna(
    train_ts,
    valid_ts,
    transform,
    experiment_name='nbeats_optimization',
    n_trials=50,
    #n_epochs=100,
    #random_seed=42
    default_params:dict=None
):
    # Función objetivo para Optuna
    def objective(trial):
        input_chunk_length = trial.suggest_int('input_chunk_length', 20, 60)
        output_chunk_length = trial.suggest_int('output_chunk_length', 12, 30)
        num_stacks = trial.suggest_int('num_stacks', 1, 3)
        num_blocks = trial.suggest_int('num_blocks', 1, 5)
        num_layers = trial.suggest_int('num_layers', 2, 6)
        dropout = trial.suggest_uniform('dropout', 0.0, 0)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)

        model_name = f"{experiment_name}_trial_{trial.number}"
        trial_params = {
            'input_chunk_length': input_chunk_length,
            'output_chunk_length': output_chunk_length,
            'num_blocks': num_blocks,
            'num_blocks': num_blocks,
            'num_layers': num_layers,
            'dropout': dropout,
            'learning_rate': learning_rate,
            #'optimizer_kwargs': {'lr': learning_rate},
            #'n_epochs': n_epochs,
            #'random_state': random_seed,
            # Model Name
            'model_name': model_name,
            'force_reset':True,
            'save_checkpoints':True,
        }
        if default_params:
          trial_params = {**default_params, **trial_params}
        
        # Evaluación con historical_forecasts
        #model = train_nbeats_model(series_scaled, input_chunk_length, output_chunk_length, num_blocks, thetas_dim, hidden_layer_units, dropout, learning_rate)
        results = train_nbeats(
            train_ts,
            valid_ts,
            trial_params,
            transform
        )
        scores = results['metrics']
        error = scores['rmse']

        return error

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial), n_trials=n_trials)

    return {
        'best_params': study.best_params,
        'results': study.trials
    }


def main():
    # Create parser
    parser = argparse.ArgumentParser(description="Programa que recibe tres parámetros: fichero de datos, tipo de algoritmo y directorio de salida.")
    
    # Definir los argumentos obligatorios
    parser.add_argument('config', type=str, help="El fichero de datos (config) que se va a procesar")

    # Parsear los argumentos
    args = parser.parse_args()

    # Read raw data
    # Verificar si el fichero de datos existe
    if not os.path.isfile(args.config):
        print(f"Error: File {args.config} doesn't exist.")
        return
    # Read YAML file
    with open(args.config, 'r') as f:
        training_config = yaml.safe_load(f)

    # Setting
    RAW_DATASET_DIR = training_config["data_config"]["train_data"]["dir"]
    RAW_DIR    = training_config["results"]["raw_dir"]
    PREPRO_DIR = training_config["results"]["prepro_dir"]
    MODELS_DIR = training_config["results"]["models_dir"]

    # Training
    TRAINING_MODELS_CONFIG = training_config["models"]

    # Create directories
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PREPRO_DIR, exist_ok=True)

    
    # Check if existe raw directory
    # cd data/raw
    # unzip air-quality-madrid.zip -d air-quality-madrid
    if not os.path.isdir(RAW_DATASET_DIR):
        print(f"Error: Directory {RAW_DATASET_DIR} doesn't exist.")
        return -1

    dataset_name = training_config["data_config"]["train_data"]["dataset_name"]
    logger.info(f"dataset_name: {dataset_name}")
    
    # Read raw data
    now = time.time()
    raw_data = read_air_quality_dataset(RAW_DATASET_DIR)
    later = time.time()
    logger.info(f"[DATA-WRANGLING] Time: {later - now}")

    # Run Data Wrangling
    now = time.time()
    prepro_data = run_data_wrangling(raw_data)
    later = time.time()
    logger.info(f"[DATA-WRANGLING] Time: {later - now}")
    stationary_features = prepro_data['stationary_cols']
    logger.info(f"[DATA-WRANGLING] Stationary: {stationary_features}")
    no_stationary_features = prepro_data['no_stationary_cols']
    logger.info(f"[DATA-WRANGLING] No Stationary: {no_stationary_features}")
    prepro_daily_data = prepro_data['series']
    prepro_stat_daily_data = prepro_daily_data[stationary_features]
    prepro_stat_daily_data.to_csv(os.path.join(PREPRO_DIR, f'{dataset_name}_clean.csv'))

    # Partition data
    now = time.time()
    partition_ts_dict = run_partition_data(prepro_stat_daily_data, train_ratio=0.8, val_ratio=0.1)
    later = time.time()
    logger.info(f"[DATA-PARTITION] Time: {later - now}")
    for ts_name, (train_ts, valid_ts, test_ts) in partition_ts_dict.items():
        train_ts.to_csv(os.path.join(PREPRO_DIR, f'{dataset_name}_{ts_name}_train.csv'))
        valid_ts.to_csv(os.path.join(PREPRO_DIR, f'{dataset_name}_{ts_name}_valid.csv'))
        test_ts.to_csv(os.path.join(PREPRO_DIR, f'{dataset_name}_{ts_name}_test.csv'))

    # Preprocessing data
    now = time.time()
    partition_ts_scaled_dict = run_transformation(partition_ts_dict)
    later = time.time()
    logger.info(f"[DATA-TRANSFORMATION] Time: {later - now}")
    for ts_name, (transform, train_ts, valid_ts, test_ts) in partition_ts_scaled_dict.items():
        train_ts.to_csv(os.path.join(PREPRO_DIR, f'{dataset_name}_{ts_name}_train.csv'))
        valid_ts.to_csv(os.path.join(PREPRO_DIR, f'{dataset_name}_{ts_name}_valid.csv'))
        test_ts.to_csv(os.path.join(PREPRO_DIR, f'{dataset_name}_{ts_name}_test.csv'))
        with open(os.path.join(PREPRO_DIR, f'{dataset_name}_{ts_name}_transform.pkl'), "wb") as f:
            pickle.dump(transform, f)
    
    
    for model_config in TRAINING_MODELS_CONFIG:
        # Get data
        ts_name = 'total'
        transform, train_ts_scaled, valid_ts_scaled, test_ts_scaled = partition_ts_scaled_dict['total']

        # ================================
        # Train Base Model
        # ================================

        # Read Setting
        logger.info(f"[TRAINING] Config: {model_config}")
        base_model_name = model_config['baseline']['model_name']
        train_caller = get_caller(model_config['baseline']['train_caller'])
        default_params = read_model_parms(model_config['baseline']['default_params_path'])
        base_model_results_dir = model_config['baseline']['output_dir']

        # Run Base Model Training
        run_traininig(
            train_ts_scaled,
            valid_ts_scaled,
            test_ts_scaled,
            trainer=train_caller,
            model_params=default_params,
            model_name=base_model_name,
            models_dir=base_model_results_dir,
            transform=transform
        )
        
        # ================================
        # HPO Model
        # ================================
        
        # Read Setting
        hpo_model_name = model_config['hpo']['model_name']
        hpo_caller = get_caller(model_config['hpo']['hpo_caller'])
        hpo_n_trials = model_config['hpo']['n_trials']
        #hpo_random_seed = model_config['hpo']['random_seed']
        hpo_default_params = read_model_parms(model_config['hpo']['default_params_path'])
        hpo_model_results_dir = model_config['hpo']['output_dir']

        # Run optimization
        run_optimization(
            train_ts_scaled,
            valid_ts_scaled,
            optimizer=hpo_caller,
            default_params=hpo_default_params,
            model_name=hpo_model_name,
            models_dir=hpo_model_results_dir,
            transform=transform,
            n_trials=hpo_n_trials
        )

        # ================================
        # Train Best Model
        # ================================

        # Read Best Parameters
        best_model_name = model_config['best_model']['model_name']
        train_caller = get_caller(model_config['baseline']['train_caller'])
        best_params = build_model_params(
            model_config['best_model']['optimized_params_path'], 
            read_model_parms(model_config['best_model']['default_params_path'])
        )
        best_model_results_dir = model_config['best_model']['output_dir']

        # Run Best Model Training
        run_traininig(
            train_ts_scaled,
            valid_ts_scaled,
            test_ts_scaled,
            trainer=train_caller,
            model_params=best_params,
            model_name=best_model_name,
            models_dir=best_model_results_dir,
            transform=transform
        )


if __name__ == "__main__":
    main()

