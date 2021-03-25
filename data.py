# Import necessities
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

class Dataset():
    def __init__(self, dataset, feature_cols, label_cols, timestep=14, 
                        batch_size=1, test_size=0.03):
        ''' Initialize the dataset
        :param dataset: Dataframe of the dataset
        :param feature_cols: list of feature names,
            e.g. ['dates', 'vaccinations'] means using dates and vaccinations as features
        :param label_cols: list of prediction targets,
            e.g. ['confirmed'] means using confirmed data as labels
        :param timestep: timestep in LSTM model
        :param test_size: the ratio of test data in the dataset
        '''
        self.dataset = dataset
        self.feature_cols = feature_cols
        self.label_cols = label_cols
        self.timestep = timestep
        self.batch_size = batch_size
        self.test_size = test_size
        # Normalize the dataset using MinMaxScaler before training
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.dataset.loc[:, dataset.columns != 'dates'] = self.scaler.fit_transform(dataset.loc[:, dataset.columns != 'dates'])
        # Split features and labels
        self.features = self.dataset[feature_cols]
        self.labels = self.dataset[label_cols]
        # Split the dataset
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                self.features, self.labels, test_size=self.test_size, shuffle=False)
    
    def get_training_set(self):
        ''' Return the time-series generator used for training
        '''
        return TimeseriesGenerator(self.x_train.to_numpy(), self.y_train.to_numpy(),
                length=self.timestep, batch_size=self.batch_size)
    
    def get_test_set(self):
        ''' Return the time-series generator used for testing
        '''
        # Add overlapping points to test set
        x_test = pd.concat([self.x_train[-self.timestep:], self.x_test])
        y_test = pd.concat([self.y_train[-self.timestep:], self.y_test])
        return TimeseriesGenerator(x_test.to_numpy(), y_test.to_numpy(),
                length=self.timestep, batch_size=self.batch_size)

def get_dataset(features, labels):
    ''' Construct the dataset with given features and labels
    '''
    # Reading in the dataset
    dataset_csv = pd.read_csv('./data/dataset.csv', index_col=0)
    dataset_rolling_csv = pd.read_csv('./data/dataset_rolling.csv', index_col=0)
    dataset = Dataset(dataset_csv, feature_cols=features, label_cols=labels)
    dataset_rolling = Dataset(dataset_rolling_csv, feature_cols=features, label_cols=labels)
    return dataset, dataset_rolling

def get_dataset_confirmed():
    ''' Construct the dataset with confirmed cases as labels
    '''
    return get_dataset(features = [
            'dates', 'people_vaccinated', 'mobility', 'hosp_patients', 'icu_patients', 'total_tests', 'median_age',
        ], labels = ['confirmed'])

def get_dataset_trends():
    ''' Construct the dataset with search trends as labels
    '''
    return get_dataset(features = [
        'dates', 'people_vaccinated', 'mobility', 'hosp_patients', 'icu_patients', 'total_tests', 'median_age',
    ], labels = [
        'trends_vaccine', 'trends_covid',
    ])
    # features = ['dates', 'people_vaccinated', 'confirmed', 'deaths', 'deaths_increment', 'total_tests', 'median_age']

def get_dataset_covid():
    ''' Construct the dataset with 3 groups of COVID cases
    '''
    return get_dataset(features = [
            'dates', 'people_vaccinated', 'mobility', 'hosp_patients', 'icu_patients', 'total_tests', 'median_age',
        ], labels = [
            'confirmed', 'recovered', 'deaths',
        ])