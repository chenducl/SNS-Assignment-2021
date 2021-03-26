## COVID-19 Forcasting

```
SNS Assignment - UCL 2021
Student Number: 20159401
```

***

## Introduction

In this assignment, I implement a COVID-19 cases and Google Trends forecasting program in Python. The dataset is constructed with diverse data sources, ranging from medical care, mobility, vaccination, etc.

This repository is composed of:

- The codes for `preprocessing` the original data
- The `dataset` I construct, containing 19 features extracted from different sources, published in CSV formats
- The codes for `training` LSTM and transformer models, and parameter `tuning` algorithms as well

## Data Sources

- Medical care, vaccination, and hospital data: [OWID](https://github.com/owid/covid-19-data)
- Google search trends for COVID keywords: [Google Trends](https://trends.google.com/trends/)
- Global mobility data: [Apple Mobility](https://covid19.apple.com/mobility)
- Global COVID-19 cases data: [JHU CSSE](https://github.com/CSSEGISandData/COVID-19)

## Code Structures

Code sources for underlying class definitions and reusable components:

```
- data.py          ...................    Dataset class definition
- base_model.py    ...................    BaseModel class definition
- lstm_model.py    ...................    LSTMModel class definition
```

iPython scripts for showing experiment settings, implementations and plot results:

```
- trends_crawler.ipynb            .................  Google Trends crawler implementations
- preprocess.ipynb                .................  Data preprocessing
- lstm_experiment.ipynb           .................  LSTM model experiments
- transformer_experiment.ipynb    .................  Transformer training and plots
```

## Usage

### Dataset

The dataset I construct contains 19 features derived from diverse data sources, and is presented in `./data/dataset.csv` and `./data/dataset_rolling.csv` (involving the data rolling method) files.

``` python
Index(['dates', 'confirmed', 'deaths', 'recovered', 'unrecovered',
       'confirmed_increment', 'deaths_increment', 'recovered_increment',
       'unrecovered_increment', 'death_rate', 'recovered_rate', 'mobility',
       'people_vaccinated', 'total_tests', 'icu_patients', 'hosp_patients',
       'median_age', 'trends_covid', 'trends_vaccine'],
      dtype='object')
```

You can simply read in the CSV files with `pandas` package to get the original data, or the data after rolling operations, which is more smoothed.

```python
dataset_csv = pd.read_csv('./data/dataset.csv', index_col=0)
dataset_rolling_csv = pd.read_csv('./data/dataset_rolling.csv', index_col=0)
```

- Construct Dataset Objects

I also implement a Dataset class, to automatically read in and `scale` the data to range [0, 1] for training. You can specify the `features` and `targets` when initializing, and the dataset object will split the  samples with a given testing sample ratio.

```python
# Kinds of features you want to use
features = ['dates', 'confirmed', 'mobility', 'people_vaccinated']
# Targets you want to predict, e.g. Google search trends for "vaccine"
labels = ['trends_vaccine']
# Read in the data files
dataset_csv = pd.read_csv('./data/dataset.csv', index_col=0)
# Construct the dataset object and it will automatically scale and split the data
dataset = Dataset(dataset_csv, feature_cols=features, label_cols=labels, test_size=0.03)
```
- Obtain Training and Testing Splits

I apply `Timeseriesgenerator` to form the training and testing sets, which can be directly fed into Keras time series models. You can use `dataset.get_training_set()` and `dataset.get_test_set()` to access both splits for training and evaluations.

## Models

Pass the parameters into LSTMModel constructors to initialize the model.

```python 
configs = {
    # LSTM model type
    'model_type': 'lstm_attetion'
    # Dataset object
    'dataset': dataset,
    # Dropout values
    'dropout': dropout,
    # Output dimensions of LSTM layers
    'output_dim': output_dim,
    # Training epochs
    'epochs': 100,
    # Logging verbose
    'verbose': 0,
}
# Initalize with configs
model = LSTMModel(**configs)
```
Then, you can use `model.read_model(model_path)` or `model.contruct_model()` to select whether to read in an existing or pretrained model, or to construct an empty LSTM model.
 
```python
# Construct the model with 'model_type'
model.construct_model()
# Training
model.train()
# Evaluating
print('MSE: ', model.evaluate())
```
Use `model.train()` to fit the dataset, and `model.evaluate()` to get the evaluations on testing samples.