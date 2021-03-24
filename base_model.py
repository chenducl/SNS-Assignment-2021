# Import necessities
import os
import seaborn as sns
import pickle as pkl
from matplotlib import pyplot as plt
from six import unichr
from tensorflow.keras.models import load_model

# Set plotting style
sns.set_style('whitegrid')
sns.set_palette('Set2')

DEFAULT_CONFIGS = {
    'dataset': None,
    'model_path': '',
    'output_dim': 128,
    'epochs': 400,
    'verbose': 1,
    'loss': 'mse',
    'optimizer': 'adam',
    'dropout': 0.2,
    'share_attention': False,
}


class BaseModel():
    ''' Other models are derived by inheritance from BaseModel.
    '''

    def __init__(self, dataset, model_path, output_dim, epochs,
                 verbose, loss, optimizer, dropout, **args):
        ''' Initialize the model parameters
        param dataset: dataset Dataframe
        param output_dim: output dimension
        param epochs: training epochs
        param verbose: verbose level of logging
        param loss: loss function for training
        param optimizer: optimizer for training
        '''
        self.dataset = dataset
        self.model_path = model_path
        self.output_dim = output_dim
        self.epochs = epochs
        self.verbose = verbose
        self.loss = loss
        self.optimizer = optimizer
        self.dropout = dropout
        self.history = None

    def read_model(self):
        ''' Load model from the given path
        '''
        self.model = load_model(self.model_path)

    def train(self):
        ''' Training
        '''
        training_set = self.dataset.get_training_set()
        self.history = self.model.fit(
            training_set, epochs=self.epochs, verbose=self.verbose)
        return self.history

    def save_model(self):
        ''' Save model and training history (if has) to path
        '''
        self.model.save(self.model_path)
        if self.history != None:
            with open(f'{self.model_path}/history.pkl', 'wb') as history:
                pkl.dump(self.history.history, history)

    def predict(self, input):
        ''' Predict results of the given input
        param input: timeseries generator of the input series
        '''
        return self.model.predict(input)

    def evaluate(self):
        ''' Evaluate and return the loss of the test set
        '''
        return self.model.evaluate(self.dataset.get_test_set())

    def plot(self, title=None):
        ''' Plot the training/testing set and model predictions for comparison
        '''
        plt.figure(figsize=(10, 8))
        if title is None:
            plt.title('Labels and Predictions on ' +
                      ', '.join(self.dataset.label_cols))
        else:
            plt.title(title)
        len_train = len(self.dataset.y_train)
        len_test = len(self.dataset.y_test)
        plt.plot(range(0, len_train), self.dataset.y_train, label='y_train')
        plt.plot(range(self.dataset.timestamp, len_train), self.model.predict(
            self.dataset.get_training_set()), label='pred_train')
        plt.plot(range(len_train, len_train+len_test),
                 self.dataset.y_test, label='y_test')
        plt.plot(range(len_train, len_train+len_test),
                 self.model.predict(self.dataset.get_test_set()), label='pred_test')
        plt.legend()
        plt.show()
