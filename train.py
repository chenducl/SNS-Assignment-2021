# Import necessities
from lstm_model import *
from data import *
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

def train_activations():
    # Get confirmed cases dataset
    _, ds_rolling = get_dataset_confirmed()
    # Model types to be tested
    model_types = ['single_layer_lstm']
    # Activations to be tested
    activations = ['sigmoid', 'tanh', 'linear', 'relu']
    
    for model_type in model_types:
        for activation in activations:
            configs = {
                'dataset': ds_rolling,
                'model_path': f"./models/{model_type}_confirmed_{activation}",
                'model_type': model_type,
                'activation': activation,
            }
            print(f'Start training {model_type}')
            model = LSTMModel(**configs)
            model.construct_model()
            model.train()
            model.save_model()


def train_trends():
    # Get confirmed cases dataset
    _, ds_rolling = get_dataset_trends()
    # Model types to be tested
    model_types = ['attention_lstm']
    
    for model_type in model_types:
        configs = {
            'dataset': ds_rolling,
            'model_path': f"./models/{model_type}_trends",
            'model_type': model_type,
        }
        print(f'Start training {model_type}')
        model = LSTMModel(**configs)
        model.construct_model()
        model.train()
        model.save_model()
        

def train(dropout, output_dim):
    # Get confirmed cases dataset
    output_dim = int(output_dim)
    _, ds_rolling = get_dataset_confirmed()
    # Model types to be tested
    model_type = 'multi_layer_lstm'
    
    configs = {
        'dataset': ds_rolling,
        'model_type': model_type,
        'dropout': dropout,
        'output_dim': output_dim,
        'epochs': 10,
        'verbose': 0,
    }
    print(f'Start training {model_type}')
    model = LSTMModel(**configs)
    model.construct_model()
    model.train()
    return model.evaluate()


def bayesian_optimization():
    pbounds = {
        'dropout': (0.0, 0.20),
        'output_dim': (64, 256),
    }
    logger = JSONLogger(path="./logs.json")
    optimizer = BayesianOptimization(
        f=train,
        pbounds=pbounds,
        random_state=1,
    )
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    optimizer.maximize(
        init_points=5,
        n_iter=15,
    )

if __name__ == "__main__":
    train_trends()