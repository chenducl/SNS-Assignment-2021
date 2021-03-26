# Import necessities
from lstm_model import *
from data import *
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs


def train(dropout, output_dim):
    ''' Black-box training function
    '''
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
    ''' Bayesian Optimization process
    '''
    # Parameters range
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
        # Diversify the search space
        init_points=5,
        # Number of iterations
        n_iter=15,
    )


if __name__ == "__main__":
    bayesian_optimization()
    