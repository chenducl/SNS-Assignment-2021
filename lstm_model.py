# Import necessities
import seaborn as sns

import tensorflow.keras.backend as K
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

from base_model import BaseModel

# Set plotting style
sns.set_style('whitegrid')
sns.set_palette('Set2')

class LSTMModel(BaseModel):
    ''' LSTM models structures and implementation.
    '''
    def __init__(self, dataset=None, model_path='', model_type='',
                        output_dim=128, epochs=400, verbose=1,
                        loss='mse', optimizer='adam', dropout=0.2,
                        activation='sigmoid', share_attention=False, **args):
        ''' Initialize the model parameters
        param dataset: Dataset object
        param output_dim: output dimension
        param epochs: training epochs
        param verbose: verbose level of logging
        param loss: loss function for training
        param optimizer: optimizer for training
        '''
        super(LSTMModel, self).__init__(dataset, model_path,
                                        output_dim, epochs, verbose,
                                        loss, optimizer, dropout, **args)
        self.model_type = model_type
        self.share_attention = share_attention
        self.activation = activation
    
    def attention_block(self, input):
        ''' Attention block
        '''
        # (batch_size, time_steps, input_dim)
        input_dim = int(input.shape[2])
        # Transpose with Permute
        a = Permute((2, 1))(input)
        a = Reshape((input_dim, self.dataset.timestamp))(a)
        a = Dense(self.dataset.timestamp, activation='softmax')(a)
        if self.share_attention:
            a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
            a = RepeatVector(input_dim)(a)
        a_probs = Permute((2, 1), name='attention_vec')(a)
        return Multiply()([input, a_probs])

    def construct_single_layer_lstm(self):
        self.model = Sequential()
        self.model.add(LSTM(units=self.output_dim, input_shape=(self.dataset.timestamp, len(self.dataset.feature_cols))))
        # self.model.add(Dropout(self.dropout))
        self.model.add(Dense(units=len(self.dataset.label_cols)))
        # Activation Functions
        if self.activation != None:
            self.model.add(Activation(self.activation))
        # MSE loss and adam optimizer
        self.model.compile(loss=self.loss, optimizer=self.optimizer)

    def construct_multi_layer_lstm(self):
        self.model = Sequential()
        # Three-layer LSTM with Dropout
        self.model.add(LSTM(units=self.output_dim, return_sequences=True, input_shape=(self.dataset.timestamp, len(self.dataset.feature_cols))))
        self.model.add(Dropout(self.dropout))
        self.model.add(LSTM(units=self.output_dim, return_sequences=True))
        self.model.add(Dropout(self.dropout))
        self.model.add(LSTM(units=self.output_dim, return_sequences=False))
        self.model.add(Dropout(self.dropout))
        # Dense Layer
        self.model.add(Dense(units=len(self.dataset.label_cols)))
        # Activation Functions
        if self.activation != None:
            self.model.add(Activation(self.activation))
        # MSE loss and adam optimizer
        self.model.compile(loss=self.loss, optimizer=self.optimizer)

    def construct_attention_lstm(self):
        inputs = Input(shape=(self.dataset.timestamp, len(self.dataset.feature_cols)))
        attention = self.attention_block(input)
        attention = LSTM(units=self.output_dim, return_sequences=False)(attention)
        output = Dense(1, activation=self.activation)(attention)
        self.model = Model(input=[inputs], output=output)

    def construct_lstm_attention(self):
        # Inputs Layer
        inputs = Input(shape=(self.dataset.timestamp, len(self.dataset.feature_cols)))
        # Single LSTM Layer
        lstm = LSTM(units=self.output_dim, return_sequences=True)(inputs)
        # Attention Block
        attention = self.attention_block(lstm)
        # Flatten to connect with Dense Layer
        attention = Flatten()(attention)
        output = Dense(1, activation=self.activation)(attention)
        self.model = Model(input=[inputs], output=output)

    def construct_model(self):
        ''' Build model from scratch
            - single_layer_lstm
            - multi_layer_lstm
            - attention_lstm
            - lstm_attention
            - transformer
        '''
        if self.model_type is 'single_layer_lstm':
            self.construct_single_layer_lstm()
        elif self.model_type is 'multi_layer_lstm':
            self.construct_multi_layer_lstm()
        elif self.model_type is 'attention_lstm':
            self.construct_attention_lstm()
        elif self.model_type is 'lstm_attention':
            self.construct_lstm_attention()
        else:
            raise(f'model_type {self.model_type} Not Implemented!')

