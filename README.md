# Project: Air Quality Madrid

<p align="center">
  <img src="images/banner.png">
</p>


## Dataset

In this section, the League of Legends dataset selected for this work is described. To better understand the content of the dataset, a brief introduction to how League of Legends works is provided. Finally, the structure of this dataset is explained.

### Description

The complete list of possible measurements and their explanations (following the original explanation document) are:

* **SO_2**: sulphur dioxide level measured in μg/m³. High levels of sulphur dioxide can produce irritation in the skin and membranes, and worsen asthma or heart diseases in sensitive groups.
* **CO**: carbon monoxide level measured in mg/m³. Carbon monoxide poisoning involves headaches, dizziness and confusion in short exposures and can result in loss of consciousness, arrhythmias, seizures or even death in the long term.
* **NO**: nitric oxide level measured in μg/m³. This is a highly corrosive gas generated among others by motor vehicles and fuel burning processes.
* **NO_2**: nitrogen dioxide level measured in μg/m³. Long-term exposure is a cause of chronic lung diseases, and are harmful for the vegetation.
* **PM25**: particles smaller than 2.5 μm level measured in μg/m³. The size of these particles allow them to penetrate into the gas exchange regions of the lungs (alveolus) and even enter the arteries. Long-term exposure is proven to be related to low birth weight and high blood pressure in newborn babies.
* **PM10**: particles smaller than 10 μm. Even though the cannot penetrate the alveolus, they can still penetrate through the lungs and affect other organs. Long term exposure can result in lung cancer and cardiovascular complications.
* **NOx**: nitrous oxides level measured in μg/m³. Affect the human respiratory system worsening asthma or other diseases, and are responsible of the yellowish-brown color of photochemical smog.
* **O_3**: ozone level measured in μg/m³. High levels can produce asthma, bronchytis or other chronic pulmonary diseases in sensitive groups or outdoor workers.
* **TOL**: toluene (methylbenzene) level measured in μg/m³. Long-term exposure to this substance (present in tobacco smkoke as well) can result in kidney complications or permanent brain damage.
* **BEN**: benzene level measured in μg/m³. Benzene is a eye and skin irritant, and long exposures may result in several types of cancer, leukaemia and anaemias. Benzene is considered a group 1 carcinogenic to humans by the IARC.
* **EBE**: ethylbenzene level measured in μg/m³. Long term exposure can cause hearing or kidney problems and the IARC has concluded that long-term exposure can produce cancer.
* **MXY**: m-xylene level measured in μg/m³. Xylenes can affect not only air but also water and soil, and a long exposure to high levels of xylenes can result in diseases affecting the liver, kidney and nervous system (especially memory and affected stimulus reaction).
* **PXY**: p-xylene level measured in μg/m³. See MXY for xylene exposure effects on health.
* **OXY**: o-xylene level measured in μg/m³. See MXY for xylene exposure effects on health.
* **TCH**: total hydrocarbons level measured in mg/m³. This group of substances can be responsible of different blood, immune system, liver, spleen, kidneys or lung diseases.
* **CH4v: methane level measured in mg/m³. This gas is an asphyxiant, which displaces the oxygen animals need to breath. Displaced oxygen can result in dizzinnes, weakness, nausea and loss of coordination.
* **NMHC**: non-methane hydrocarbons (volatile organic compounds) level measured in mg/m³. Long exposure to some of these substances can result in damage to the liver, kidney, and central nervous system. Some of them are suspected to cause cancer in humans.

The file containing the dataset is stored in [air-quality-madrid.zip](data/raw/air-quality-madrid.zip).

The dataset is composed of annual samples from 2013 to 2017. This dataset has been compressed into a zip file. For this reason, the first step is to unzip it with the following command.

```bash
cd data/raw/
unzip air-quality-madrid.zip  -d air-quality-madrid

Archive:  air-quality-madrid.zip
   creating: air-quality-madrid/csvs_per_year/
   creating: air-quality-madrid/data/
  inflating: air-quality-madrid/stations.csv  
   creating: air-quality-madrid/csvs_per_year/csvs_per_year/
  inflating: air-quality-madrid/csvs_per_year/csvs_per_year/madrid_2013.csv  
  inflating: air-quality-madrid/csvs_per_year/csvs_per_year/madrid_2004.csv  
  inflating: air-quality-madrid/csvs_per_year/csvs_per_year/madrid_2002.csv  
  inflating: air-quality-madrid/csvs_per_year/csvs_per_year/madrid_2014.csv  
  inflating: air-quality-madrid/csvs_per_year/csvs_per_year/madrid_2015.csv  
  inflating: air-quality-madrid/csvs_per_year/csvs_per_year/madrid_2006.csv  
  inflating: air-quality-madrid/csvs_per_year/csvs_per_year/madrid_2001.csv  
  inflating: air-quality-madrid/csvs_per_year/csvs_per_year/madrid_2009.csv  
  inflating: air-quality-madrid/csvs_per_year/csvs_per_year/madrid_2016.csv  
  inflating: air-quality-madrid/csvs_per_year/csvs_per_year/madrid_2003.csv  
  inflating: air-quality-madrid/csvs_per_year/csvs_per_year/madrid_2005.csv  
  inflating: air-quality-madrid/csvs_per_year/csvs_per_year/madrid_2011.csv  
  inflating: air-quality-madrid/csvs_per_year/csvs_per_year/madrid_2012.csv  
  inflating: air-quality-madrid/csvs_per_year/csvs_per_year/madrid_2010.csv  
  inflating: air-quality-madrid/csvs_per_year/csvs_per_year/madrid_2008.csv  
  inflating: air-quality-madrid/csvs_per_year/csvs_per_year/madrid_2018.csv  
  inflating: air-quality-madrid/csvs_per_year/csvs_per_year/madrid_2007.csv  
  inflating: air-quality-madrid/csvs_per_year/csvs_per_year/madrid_2017.csv  
```

```bash
ls -lh data/raw/air-quality-madrid/csvs_per_year/csvs_per_year/
total 498M
-rw-r--r-- 1 codespace codespace  38M Jan 31 19:06 madrid_2001.csv
-rw-r--r-- 1 codespace codespace  38M Jan 31 19:06 madrid_2002.csv
-rw-r--r-- 1 codespace codespace  42M Jan 31 19:06 madrid_2003.csv
-rw-r--r-- 1 codespace codespace  43M Jan 31 19:06 madrid_2004.csv
-rw-r--r-- 1 codespace codespace  41M Jan 31 19:06 madrid_2005.csv
-rw-r--r-- 1 codespace codespace  41M Jan 31 19:06 madrid_2006.csv
-rw-r--r-- 1 codespace codespace  39M Jan 31 19:06 madrid_2007.csv
-rw-r--r-- 1 codespace codespace  39M Jan 31 19:06 madrid_2008.csv
-rw-r--r-- 1 codespace codespace  37M Jan 31 19:06 madrid_2009.csv
-rw-r--r-- 1 codespace codespace  28M Jan 31 19:06 madrid_2010.csv
-rw-r--r-- 1 codespace codespace  17M Jan 31 19:06 madrid_2011.csv
-rw-r--r-- 1 codespace codespace  16M Jan 31 19:06 madrid_2012.csv
-rw-r--r-- 1 codespace codespace  16M Jan 31 19:06 madrid_2013.csv
-rw-r--r-- 1 codespace codespace  16M Jan 31 19:06 madrid_2014.csv
-rw-r--r-- 1 codespace codespace  16M Jan 31 19:06 madrid_2015.csv
-rw-r--r-- 1 codespace codespace  16M Jan 31 19:06 madrid_2016.csv
-rw-r--r-- 1 codespace codespace  17M Jan 31 19:06 madrid_2017.csv
-rw-r--r-- 1 codespace codespace 5.7M Jan 31 19:06 madrid_2018.csv
```
## Data Wrangling

TBA

# Partion Data

The dataset has a size of 9879 rows and 40 columns. The raw data is divided into two subsets: training and testing. With the following command you can see these partitions:

```bash
ls -l data/prepro/
ls -l data/raw/

total 2832
-rw-rw-r-- 1 aztleclan aztleclan 1446502 abr 13  2020 high_diamond_ranked_10min.csv
-rw-rw-r-- 1 aztleclan aztleclan  289685 nov 26 01:08 high_diamond_ranked_10min_raw_test.csv
-rw-rw-r-- 1 aztleclan aztleclan 1157422 nov 26 01:08 high_diamond_ranked_10min_raw_train.csv
```

On the other hand, there is the preprocessed data that was obtained from the raw data. The preprocessed dataset is partitioned into three subsets: train (60%), valid (20%) and test (60). 

| Conjunto   | Size         |
|----------- |--------------|
| full train | 7903 (0.80)% | 
| train      | 5927 (0.60)% |
| valid      | 1976 (0.20)% |
| test       | 1976 (0.20)% |

With the following command you can see these partitions:

```bash
ls -l data/prepro/
total 1504
-rw-rw-r-- 1 aztleclan aztleclan 450909 nov 26 01:08 high_diamond_ranked_10min_full_train_clean.csv
-rw-rw-r-- 1 aztleclan aztleclan 111599 nov 26 01:08 high_diamond_ranked_10min_test_clean.csv
-rw-rw-r-- 1 aztleclan aztleclan 338080 nov 26 01:08 high_diamond_ranked_10min_train_clean.csv
-rw-rw-r-- 1 aztleclan aztleclan 111927 nov 26 01:08 high_diamond_ranked_10min_valid_clean.csv
```

## Technologies

- Python 3.10.12
- [Flask 3.0.3](https://flask.palletsprojects.com/en/stable/) is a lightweight and flexible web framework for Python that allows developers to build web applications quickly and with minimal overhead. It provides essential tools for routing, templating, and handling HTTP requests, while allowing for easy extensibility through a wide range of plugins and extensions.
- [PyTorch 2.5.1](https://pytorch.org/) is an open-source deep learning framework developed by Facebook AI. It provides a flexible and dynamic computation graph, making it easy to build and modify neural networks. The library supports automatic differentiation (autograd), GPU acceleration, and a rich ecosystem for model training and deployment. PyTorch integrates well with libraries like Torchvision for computer vision and TorchText for NLP. It is widely used in both research and production due to its ease of use and strong community support.
- [Optuna 4.2.0](https://optuna.org/) is an open-source hyperparameter optimization framework designed for machine learning and deep learning models. It uses an efficient, automated search strategy based on Bayesian optimization, Tree-structured Parzen Estimator (TPE), and other algorithms to find the best hyperparameters. The library features a flexible and lightweight API, allowing users to define optimization objectives with minimal code. It supports pruning of unpromising trials, parallel and distributed optimization, and visualization of optimization results. Optuna seamlessly integrates with popular ML frameworks like TensorFlow, PyTorch, XGBoost, and LightGBM.
- [Darts 0.32.0](https://unit8co.github.io/darts/) is a Python library for time series forecasting and analysis. It provides a unified interface for multiple forecasting models, including statistical methods (ARIMA, Exponential Smoothing) and deep learning-based models (RNNs, TCN, Transformer). The library supports missing value handling, probabilistic forecasting, and ensembling techniques. It integrates seamlessly with PyTorch and LightGBM, allowing users to train custom models with ease. Darts also includes built-in utilities for data preprocessing, evaluation metrics, and backtesting.
- [TensorBoard](https://www.tensorflow.org/tensorboard) is a visualization toolkit for TensorFlow that helps monitor and analyze machine learning experiments. It provides interactive dashboards for tracking metrics like loss and accuracy, visualizing computational graphs, and inspecting model parameters. The library supports logging scalars, histograms, images, and embeddings to gain insights into training performance. TensorBoard integrates seamlessly with TensorFlow but can also be used with PyTorch and other frameworks through custom logging. It enables real-time experiment tracking, making it easier to debug and optimize models.
- [Pipenv](https://pipenv.pypa.io/en/latest/) is a tool for managing dependencies in Python projects, combining the functionalities of pip and virtualenv.
- [Jupyter](https://jupyter.org/) is an open-source web application that allows users to create and share documents containing live code, equations, visualizations, and narrative text. It supports various programming languages, including Python, R, and Julia, making it widely used for data science, machine learning, and academic research. Its interactive environment enables real-time code execution, making it an essential tool for exploratory data analysis and visualization.



## Architecture

The application code is composed of the following components:

- [`app.py`](app.py) - Module with Flask application.
- [`train.py`](train.py) - Module for preprocessing, feature selection and training with XGBoost and Random Forest models.
- [`predict.py`](predict.py) - Module to obtain predictions with the test subset for the XGBoost and Random Forest models.
- [`test_app.py`](test_app.py) - Module to test the Flask application from the test subset.
- [`Dockerfile`](Dockerfile) - Dockerfile to build an image for the Flask application that returns predictions for the XGBoost and Random Forest models.
- [`docker-compose`](docker-compose.yaml) - Docker compose serving the Flask application on port 5000.


The configuration for the application is in the [`config/`](config/) folder:

- [`app_config.yaml`](config/app_config.yaml) - Flask Application configuration data
- [`train_config.yaml`](config/train_config.yaml)  - Configuration data for Training.
- [`predict_config.yaml`](config/predict_config.yaml)  - Configuration data for Testing.
- [`test_app_config.yaml`](config/test_app_config.yaml)  - Configuration data for App Flask Testing.

Log files are stored in the [logs](logs) directory.

## Preparation

For dependency management, we use pipenv, so you need to install it:

```bash
pip install pipenv
```

Once installed, you can install the app dependencies:

```bash
pipenv install --dev
```

## Running the application


### Running with Docker-Compose

The easiest way to run the application is with `docker-compose`. 

First we build the docker image.

```bash
docker-compose build
```

Then, we boot the image with the following command:

```bash
docker-compose up -d
```

### Running with Docker (without compose)

Sometimes you might want to run the application in
Docker without Docker Compose, e.g., for debugging purposes.

First, prepare the environment by running Docker Compose
as in the previous section.

Next, build the image:

```bash
docker build -t ml-zoomcamp-capstone-air-quality:3.12-slim . 
```
Run it:

```bash
docker run -it --rm \
    -p 5000:5000 \
    ml-zoomcamp-capstone-air-quality:3.12-slim
```

### Running Flask 

We can also start the Flask application as a Python application with the following command.

```bash
pipenv shell

python app.py 
```

La aplicación flask [app.py](app.py) tiene un fichero de configuración donde se especifica las series temporales y los modelos de referencia y optimizado entrenados usando los modelos **TCN** y **NBEATS** de la librería **darts**.
This module requires the configuration file [app_config.yaml](config/app_config.yaml) with the following parameters:

- `data_config[*].transform_path`: Model path with transformation operations applied to the training subset.
- `data_config[*].train_path`: Training subset path for the **total** time series.
- `data_config[*].valid_path`: Validation subset path for the **total** time series.
- `data_config[*].test_path`: Test subset path for the **total** time series.
- `models[*].model_name`: Model name of the **darts** library. Valid values: **NBeats** and **TCN**.
- `models[*].experiment_name`: Experiment name of the **darts** library.
- `models[*].model_path`: Path of the model trained using the darts library.
- `results.prediction_dir`: Directory where the predictions obtained from the test subset are stored.

```yaml
data_config:
  - series_name: total
    transform_path : data/prepro/air-quality-madrid_total_transform.pkl
    train_path: data/prepro/air-quality-madrid_total_train.csv
    valid_path: data/prepro/air-quality-madrid_total_valid.csv
    test_path : data/prepro/air-quality-madrid_total_test.csv


models:
  - model_name: TCN  
    model_path: models/tcn/baseline_model_tcn.pt
    experiment_name: baseline_tcn
  - model_name: TCN
    model_path: models/tcn/best_model_tcn.pt
    experiment_name: optimized_tcn
  - model_name: NBeats
    model_path: models/nbeats/baseline_model_nbeats.pt
    experiment_name: baseline_nbeats
  - model_name: NBeats
    model_path: models/nbeats/best_model_nbeats.pt
    experiment_name: optimized_nbeats
```


## Experiments

For experiments, we use Jupyter notebooks.
They are in the [`notebooks`](notebooks/) folder.

To start Jupyter, run:

```bash
pipenv shell

cd notebooks

jupyter notebook
```

We have the following notebooks:

- [`notebook.ipynb`](notebooks/notebook.ipynb): Notebook for training, evaluation and optimization with the XGBoost and Random Forest models.

## Training
The [train.py](train.py) module contains the logic to perform the preprocessing of the raw data, feature selection, training and optimization of the **TCN** and **NBeats** models. This module requires the configuration file [train_config.yaml](config/train_config.yaml) with the following parameters:

- `data_config.features.target`: Features to predict in the dataset.
- `data_config.train_data.path`: File with the dataset.
- `models[*]`: Darts Model Configuration.
- `models[*].model_name`: Model name of the "darts" library. Valid values: **NBeats** and **TCN**.
- `models[*].baseline.experiment_name`: The experiment name for training the base model.
- `models[*].baseline.default_params_path`: Path of the json file containing the default parameters of the model.
- `models[*].baseline.output_dir`: Directory where the training results such as the model (*.pt) are saved.
- `models[*].hpo.experiment_name`: The experiment name for HPO.
- `models[*].hpo.default_params_path`: Path of the json file containing the default parameters of the model.
- `models[*].hpo.output_dir`: Directory where the HPO results such as the optimized parameters are saved.
- `models[*].best_model.experiment_name`: The experiment name for training the optimized model.
- `models[*].best_model.default_params_path`: Path of the json file containing the default parameters of the model.
- `models[*].best_model.optimized_params_path`: Path of the json file containing the optimized model parameters.
- `models[*].best_model.output_dir`: Directory where the training results such as the optimized model (*.pt) are saved.
- `results.raw_dir`: Directory where the unpreprocessed dataset are stored.
- `results:prepro_dir`: Directory where the preprocessed subsets are stored.

The following describes the configuration file [train_config.yaml](config/train_config.yaml)

```yaml
data_config:
  train_data:
      dataset_name: air-quality-madrid
      dir: data/raw/air-quality-madrid/csvs_per_year/csvs_per_year

models:
  - model_name: TCN
    baseline: 
        experiment_name: baseline_model_tcn
        default_params_path: models/tcn/baseline_model_tcn_default_params.json
        output_dir: models/tcn
    hpo:
        experiment_name: hpo_model_tcn
        n_trials: 2
        default_params_path : models/tcn/hpo_model_tcn_default_params.json
        output_dir: models/tcn
    best_model:
        experiment_name: best_model_tcn
        default_params_path : models/tcn/hpo_model_tcn_default_params.json
        optimized_params_path: models/tcn/hpo_model_tcn_optimized_params.json
        output_dir: models/tcn

  - model_name: NBeats
    baseline: 
        experiment_name: baseline_model_nbeats
        default_params_path: models/nbeats/baseline_model_nbeats_default_params.json
        output_dir: models/nbeats
    hpo:
        experiment_name: hpo_model_nbeats
        n_trials: 2
        default_params_path : models/nbeats/hpo_model_nbeats_default_params.json
        output_dir: models/nbeats
    best_model:
        experiment_name: best_model_nbeats
        default_params_path : models/nbeats/hpo_model_nbeats_default_params.json
        optimized_params_path: models/nbeats/hpo_model_nbeats_optimized_params.json
        output_dir: models/nbeats

outputs:
  raw_dir: "data/raw"
  prepro_dir: "data/prepro"
```
During the training phase, models are generated for `TCN` and `NBEATS`. For both models `TCN` and `NBEATS`, first a trained baseline model is generated with default parameters. Then, an HPO phase is performed with the Optuna library and optimized parameters for the model are obtained using a Bayesian search. Finally, an optimized model is trained using the optimized parameters obtained in the HPO phase. These generated models are saved in the `models` directory and their content is shown below.

```bash
ls -lh models/tcn/
total 21M
-rw-rw-rw- 1 codespace root 444K Feb  3 12:44 baseline_model_tcn.pt
-rw------- 1 codespace root  23K Feb  3 12:44 baseline_model_tcn.pt.ckpt
-rw-rw-rw- 1 codespace root  367 Feb  3 11:18 baseline_model_tcn_default_params.json
-rw-rw-rw- 1 codespace root   83 Feb  3 12:44 baseline_model_tcn_eval_test.json
-rw-rw-rw- 1 codespace root   84 Feb  3 12:44 baseline_model_tcn_eval_valid.json
-rw-rw-rw- 1 codespace root  20K Feb  3 12:44 baseline_model_tcn_forecast_test.csv
-rw-rw-rw- 1 codespace root  18K Feb  3 12:44 baseline_model_tcn_forecast_test_back.csv
-rw-rw-rw- 1 codespace root  20K Feb  3 12:44 baseline_model_tcn_forecast_valid.csv
-rw-rw-rw- 1 codespace root  18K Feb  3 12:44 baseline_model_tcn_forecast_valid_back.csv
-rw-rw-rw- 1 codespace root  11M Feb  3 12:44 best_model_tcn.pt
-rw------- 1 codespace root 9.9M Feb  3 12:44 best_model_tcn.pt.ckpt
-rw-rw-rw- 1 codespace root   84 Feb  3 12:44 best_model_tcn_eval_test.json
-rw-rw-rw- 1 codespace root   85 Feb  3 12:44 best_model_tcn_eval_valid.json
-rw-rw-rw- 1 codespace root  20K Feb  3 12:44 best_model_tcn_forecast_test.csv
-rw-rw-rw- 1 codespace root  18K Feb  3 12:44 best_model_tcn_forecast_test_back.csv
-rw-rw-rw- 1 codespace root  20K Feb  3 12:44 best_model_tcn_forecast_valid.csv
-rw-rw-rw- 1 codespace root  18K Feb  3 12:44 best_model_tcn_forecast_valid_back.csv
-rw-rw-rw- 1 codespace root  101 Feb  3 11:18 hpo_model_tcn_default_params.json
-rw-rw-rw- 1 codespace root  264 Feb  3 12:44 hpo_model_tcn_optimized_params.json
-rw-rw-rw- 1 codespace root  263 Feb  3 11:18 hpo_model_tcn_params.json
```

```bash
ls -lh models/nbeats/
total 128M
-rw-rw-rw- 1 codespace root 48M Feb  3 12:47 baseline_model_nbeats.pt
-rw------- 1 codespace root 48M Feb  3 12:47 baseline_model_nbeats.pt.ckpt
-rw-rw-rw- 1 codespace root 162 Feb  3 11:18 baseline_model_nbeats_default_params.json
-rw-rw-rw- 1 codespace root  85 Feb  3 12:47 baseline_model_nbeats_eval_test.json
-rw-rw-rw- 1 codespace root  85 Feb  3 12:47 baseline_model_nbeats_eval_valid.json
-rw-rw-rw- 1 codespace root 19K Feb  3 12:47 baseline_model_nbeats_forecast_test.csv
-rw-rw-rw- 1 codespace root 19K Feb  3 12:47 baseline_model_nbeats_forecast_test_back.csv
-rw-rw-rw- 1 codespace root 19K Feb  3 12:47 baseline_model_nbeats_forecast_valid.csv
-rw-rw-rw- 1 codespace root 19K Feb  3 12:47 baseline_model_nbeats_forecast_valid_back.csv
-rw-rw-rw- 1 codespace root 17M Feb  3 13:06 best_model_nbeats.pt
-rw------- 1 codespace root 16M Feb  3 13:06 best_model_nbeats.pt.ckpt
-rw-rw-rw- 1 codespace root  85 Feb  3 13:06 best_model_nbeats_eval_test.json
-rw-rw-rw- 1 codespace root  84 Feb  3 13:06 best_model_nbeats_eval_valid.json
-rw-rw-rw- 1 codespace root 19K Feb  3 13:06 best_model_nbeats_forecast_test.csv
-rw-rw-rw- 1 codespace root 19K Feb  3 13:06 best_model_nbeats_forecast_test_back.csv
-rw-rw-rw- 1 codespace root 19K Feb  3 13:06 best_model_nbeats_forecast_valid.csv
-rw-rw-rw- 1 codespace root 19K Feb  3 13:06 best_model_nbeats_forecast_valid_back.csv
-rw-rw-rw- 1 codespace root 100 Feb  3 11:18 hpo_model_nbeats_default_params.json
-rw-rw-rw- 1 codespace root 243 Feb  3 13:05 hpo_model_nbeats_optimized_params.json
```

- [baseline_model_tcn.pt](models/tcn/baseline_model_tcn.pt): Baseline model trained using the TCN model.
- [baseline_model_tcn_default_params.json](models/tcn/baseline_model_tcn_default_params.json): TCN model default parameters used during training of the reference model.
- [baseline_model_tcn_eval_valid.json](models/nbeats/baseline_model_tcn_eval_valid.json): Evaluation metrics of the **validation subset** for the baseline model using **TCN**.
- [baseline_model_tcn_eval_test.json](models/nbeats/baseline_model_tcn_eval_test.json): Evaluation metrics of the **test subset** for the baseline model using **TCN**.
- [hpo_model_tcn_default_params.json](models/tcn/hpo_model_tcn_default_params.json): TCN model defect parameters used during the HPO phase with Optuna.
- [hpo_model_tcn_optimized_params.json](models/tcn/hpo_model_tcn_optimized_params.json): Optimized parameters obtained during the HPO phase using Optuna for the **TCN** model.
- [best_model_tcn.pt](models/tcn/best_model_tcn.pt): Optimized model trained with the optimized parameters (HPO) using the **TCN** model.
- [best_model_tcn_eval_valid.json](models/nbeats/best_model_tcn_eval_valid.json): Evaluation metrics of the **validation subset** for the optimized model using **TCN**.
- [best_model_tcn_eval_test.json](models/nbeats/best_model_tcn_eval_test.json): Evaluation metrics of the **test subset** for the optimized model using **TCN**.
- [baseline_model_nbeats.pt](models/nbeats/baseline_model_nbeats.pt): Baseline model trained using the **NBEATS**.
- [baseline_model_nbeats_default_params.json](models/nbeats/baseline_model_nbeats_default_params.json): **NBEATS** model default parameters used during training of the reference model.
- [baseline_model_nbeats_eval_valid.json](models/nbeats/baseline_model_nbeats_eval_test.json): Evaluation metrics of the **validation subset** for the baseline model using **NBEATS**.
- [baseline_model_nbeats_eval_test.json](models/nbeats/baseline_model_nbeats_eval_test.json): Evaluation metrics of the **test subset** for the baseline model using **NBEATS**.
- [hpo_model_nbeats_default_params.json](models/nbeats/hpo_model_nbeats_default_params.json): NBEATS model defect parameters used during the HPO phase with Optuna.
- [hpo_model_nbeats_optimized_params.json](models/nbeats/hpo_model_nbeats_optimized_params.json): Optimized parameters obtained during the HPO phase using Optuna for the **NBEATS** model.
- [best_model_nbeats.pt](models/nbeats/best_model_nbeats.pt): Optimized model trained with the optimized parameters (HPO) using the **NBEATS** model.
- [best_model_nbeats_eval_valid.json](models/nbeats/best_model_nbeats_eval_valid.json): Evaluation metrics of the **validation subset** for the optimized model using **NBEATS**.
- [best_model_nbeats_eval_test.json](models/nbeats/best_model_nbeats_eval_test.json): Evaluation metrics of the **test subset** for the optimized model using **NBEATS**.


A complete description of the **TCN** model parameters can be found [here](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tcn_model.html). The content of the optimized parameters for the **TCN** model is shown below:

```json
{
  "n_epochs": 10,
  "random_state": 42,
  "force_reset": true,
  "save_checkpoints": true,
  "input_chunk_length": 43,
  "output_chunk_length": 15,
  "num_filters": 73,
  "kernel_size": 6,
  "dropout": 0.03052046313238249,
  "batch_size": 20,
  "learning_rate": 0.0000705950138818228
}
```

A complete description of the **NBEATS** model parameters can be found [here](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.nbeats.html). The content of the optimized parameters for the **NBEATS** model is shown below:

```json
{
  "n_epochs": 5,
  "random_state": 42,
  "force_reset": true,
  "save_checkpoints": true,
  "input_chunk_length": 36,
  "output_chunk_length": 13,
  "num_stacks": 2,
  "num_blocks": 3,
  "num_layers": 3,
  "dropout": 0,
  "learning_rate": 0.002490575642970675
}
```

Below are the metrics results for the base and optimized *TCN* and *NBEATS*:

| Model Name         | MAE      | RMSE     | MAPE     |
|--------------------|----------|----------|----------|
| Baseline TCN       | 0.803    |          |          |
| Optimized TCN      | 0.809    |          |          |
| XBaseline NBEATS   | 0.802    |          |          |
| Optimized NBEATS   | 0.806    |          |          |

## Prediction

The [predict.py](predict.py) module contains the logic to obtain the predictions of the test subset. This module requires the configuration file [predict_config.yaml](config/predict_config.yaml) with the following parameters:

- `data_config[*].transform_path`: Model path with transformation operations applied to the training subset.
- `data_config[*].train_path`: Training subset path for the **total** time series.
- `data_config[*].valid_path`: Validation subset path for the **total** time series.
- `data_config[*].test_path`: Test subset path for the **total** time series.
- `models[*].model_name`: Model name of the **darts** library. Valid values: **NBeats** and **TCN**.
- `models[*].experiment_name`: Experiment name of the **darts** library.
- `models[*].model_path`: Path of the model trained using the darts library.
- `results.prediction_dir`: Directory where the predictions obtained from the test subset are stored.

The following describes the configuration file [predict_config.yaml](config/predict_config.yaml)

```yaml
data_config:
  - series_name: total
    transform_path : data/prepro/air-quality-madrid_total_transform.pkl
    train_path: data/prepro/air-quality-madrid_total_train.csv
    valid_path: data/prepro/air-quality-madrid_total_valid.csv
    test_path : data/prepro/air-quality-madrid_total_test.csv


models:
  - model_name: TCN  
    model_path: models/tcn/baseline_model_tcn.pt
    experiment_name: baseline_tcn
  - model_name: TCN
    model_path: models/tcn/best_model_tcn.pt
    experiment_name: optimized_tcn
  - model_name: NBeats
    model_path: models/nbeats/baseline_model_nbeats.pt
    experiment_name: baseline_nbeats
  - model_name: NBeats
    model_path: models/nbeats/best_model_nbeats.pt
    experiment_name: optimized_nbeats
    
outputs:
  prediction_dir: "output/predict"
```

To run this module, execute the following command:

```bash
pipenv shell

python predict.py config/predict_config.yaml
```
The prediction results for the different test subsets and **TCN** and **NBEATS** models are stored in the `output/predict` directory. The contents of the directory are shown below.

```bash
 ls -lh output/predict/baseline_model_tcn_total/
total 24K
-rw-rw-rw- 1 codespace codespace  84 Feb  3 14:07 baseline_model_tcn_total_eval_test.json
-rw-rw-rw- 1 codespace codespace 20K Feb  3 14:07 baseline_model_tcn_total_forecast_test.csv
```

```bash
ls -lh output/predict/best_model_tcn_total/
total 24K
-rw-rw-rw- 1 codespace codespace  84 Feb  3 14:07 best_model_tcn_total_eval_test.json
-rw-rw-rw- 1 codespace codespace 20K Feb  3 14:07 best_model_tcn_total_forecast_test.csv
```

```bash
ls -lh output/predict/baseline_model_nbeats_total/
total 24K
-rw-rw-rw- 1 codespace codespace  85 Feb  3 14:07 baseline_model_nbeats_total_eval_test.json
-rw-rw-rw- 1 codespace codespace 19K Feb  3 14:07 baseline_model_nbeats_total_forecast_test.csv
```

```bash
ls -lh output/predict/best_model_nbeats_total/
total 24K
-rw-rw-rw- 1 codespace codespace  86 Feb  3 14:07 best_model_nbeats_total_eval_test.json
-rw-rw-rw- 1 codespace codespace 19K Feb  3 14:07 best_model_nbeats_total_forecast_test.csv
```

The predictions for the test subset using the model optimized with **TCN** are stored in the `output/predict/best_model_nbeats_total` directory. As you can see, two files have been created, which are described below.

- `best_model_tcn_total_eval_test.json`. Forecasts for the test subset for the optimized NBEATS model.
- `best_model_tcn_total_forecast_test.csv`. MAPE, MAE and RMSE metrics for the test subset for the optimized NBEATS model.


The predictions for the test subset using the model optimized with **NBEATS** are stored in the `output/predict/best_model_nbeats_total` directory. As you can see, two files have been created, which are described below.

- `best_model_nbeats_total_eval_test.json`. Forecasts for the test subset for the optimized NBEATS model.
- `best_model_nbeats_total_forecast_test.csv`. MAPE, MAE and RMSE metrics for the test subset for the optimized NBEATS model.



## Test Flasks

The [app.py](app.py) module contains the logic for testing the Flask application. This module needs the configuration file [test_app_config.yaml](config/test_app_config.yaml) with the following parameters:

- `data_config[*].transform_path`: Model path with transformation operations applied to the training subset.
- `data_config[*].train_path`: Training subset path for the **total** time series.
- `data_config[*].valid_path`: Validation subset path for the **total** time series.
- `data_config[*].test_path`: Test subset path for the **total** time series.
- `endpoints.predict`: URL of the Flask application's prediction method.
- `outputs.path`: Directory where the predictions returned by the Flask application are saved.

The following describes the configuration file [test_app_config.yaml](config/test_app_config.yaml)

```yaml
data_config:
  - series_name: total
    transform_path : data/prepro/air-quality-madrid_total_transform.pkl
    train_path: data/prepro/air-quality-madrid_total_train.csv
    valid_path: data/prepro/air-quality-madrid_total_valid.csv
    test_path : data/prepro/air-quality-madrid_total_test.csv

endpoints:
  predict: "http://127.0.0.1:5000/predict"

outputs:
  path: output/test_app
```

To run this module, execute the following command:

```bash
pipenv shell

python test_app.py config/test_app_config.yaml 
```
For example, for the following entry data in json format:

```json
{
    "days": 5
}
```

The Flask application returns the following results:

```json
{
    "predictions": {
        "baseline_nbeats": [
            {
                "ds": "2016-08-07",
                "y": 123.5435299960226
            },
            {
                "ds": "2016-08-08",
                "y": 131.82764891494915
            },
            {
                "ds": "2016-08-09",
                "y": 152.6554092505334
            },
            {
                "ds": "2016-08-10",
                "y": 145.14259695871215
            },
            {
                "ds": "2016-08-11",
                "y": 143.733328929921
            }
        ],
        "baseline_tcn": [
            {
                "ds": "2016-08-07",
                "y": 73.28656556533191
            },
            {
                "ds": "2016-08-08",
                "y": 50.291452247271735
            },
            {
                "ds": "2016-08-09",
                "y": 112.71765145836082
            },
            {
                "ds": "2016-08-10",
                "y": 118.0542001802411
            },
            {
                "ds": "2016-08-11",
                "y": 91.99579015233253
            }
        ],
        "optimized_nbeats": [
            {
                "ds": "2016-08-07",
                "y": 126.64161499150347
            },
            {
                "ds": "2016-08-08",
                "y": 132.15121559899276
            },
            {
                "ds": "2016-08-09",
                "y": 130.36130820768105
            },
            {
                "ds": "2016-08-10",
                "y": 138.7782332940841
            },
            {
                "ds": "2016-08-11",
                "y": 138.28603257873726
            }
        ],
        "optimized_tcn": [
            {
                "ds": "2016-08-07",
                "y": 99.61636441123454
            },
            {
                "ds": "2016-08-08",
                "y": 87.04305128810334
            },
            {
                "ds": "2016-08-09",
                "y": 79.15772784444327
            },
            {
                "ds": "2016-08-10",
                "y": 81.29823622455692
            },
            {
                "ds": "2016-08-11",
                "y": 79.59986700966545
            }
        ]
    }
}
```

It can be observed that the result has four predictions that correspond to four models:

- `baseline_tcn`: Forecast result of the baseline TCN model.
- `optimized_tcn`: Forecast result of the optimized TCN model.
- `baseline_nbeats`: Forecast result of the baseline NBEATS model.
- `optimized_nbeats`: Forecast result of the optimized NBEATS model.