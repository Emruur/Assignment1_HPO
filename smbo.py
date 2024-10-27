import ConfigSpace
import numpy as np
import typing

from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from ConfigConverter import ConfigSpaceTransformer
from sklearn.model_selection import train_test_split



import pandas as pd
from scipy.stats import norm

'''
def logit_transform(data, epsilon=1e-10):
    data = np.array(data)
    data = np.clip(data, epsilon, 1 - epsilon)
    logit_data = np.log(data / (1 - data))
    return logit_data

'''

def convert_to_dataframe(capital_phi):
    # Extract the list of dictionaries (X) and floats (y)
    X_dicts = [item[0] for item in capital_phi]
    y_values = [item[1] for item in capital_phi]

    # Create DataFrames for X and y
    X = pd.DataFrame(X_dicts)
    y = pd.DataFrame(y_values, columns=['target'])  # Optionally, you can name the target column
    
    return X, y

class SequentialModelBasedOptimization(object):

    def __init__(self, config_space, anchor_size):
        """
        Initializes empty variables for the model, the list of runs (capital R), and the incumbent
        (theta_inc being the best found hyperparameters, theta_inc_performance being the performance
        associated with it)
        """
        self.config_space= config_space
        self.config_space_transformer = ConfigSpaceTransformer(config_space)
        self.R= []
        self.anchor_size= anchor_size
        theta_inc= {}
        theta_inc_performance= 1


    def initialize(self, capital_phi: typing.List[typing.Tuple[typing.Dict, float]]) -> None:
        """
        Initializes the model with a set of initial configurations, before it can make recommendations
        which configurations are in good regions. Note that we are minimising (lower values are preferred)

        :param capital_phi: a list of tuples, each tuple being a configuration and the performance (typically,
        error rate)
        """
        capital_phi = [(config, 1 - score) for config, score in capital_phi]
        # Define the kernel

        # Define    the pipeline with scaling
        kernel = C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-2, 1e2))
        self.gp_pipeline = Pipeline([
            ('config_transform', self.config_space_transformer),
            ('gp', GaussianProcessRegressor(
                kernel=kernel, 
                n_restarts_optimizer=10))
        ])


        # Find the tuple with the maximum performance score
        best_run = max(capital_phi, key=lambda x: x[1])
        # Set the configuration with the maximum score
        self.theta_inc = best_run[0]
        self.theta_inc_performance = best_run[1]
        self.R= capital_phi

    def fit_model(self) -> None:
        """
        Fits the internal surrogate model on the complete run list.
        Splits the data into training and test sets, trains the pipeline on the training data.
        """
        # Convert self.R to X and y DataFrames
        X, y = convert_to_dataframe(self.R)
        X["anchor_size"]= self.anchor_size
        # Fit the pipeline on the training set
        self.gp_pipeline.fit(X, y)

    def select_configuration(self) -> ConfigSpace.Configuration:
        """
        Determines which configurations are good, based on the internal surrogate model.
        Note that we are minimizing the error, but the expected improvement takes into account that.
        Therefore, we are maximizing expected improvement here.

        :return: A size n vector, same size as each element representing the EI of a given
        configuration
        """
        # maximize the acquisition function by random sampling

        n_samples = 200
        samples = self.config_space.sample_configuration(n_samples)
        expected_improvements= SequentialModelBasedOptimization.expected_improvement(self.gp_pipeline, self.theta_inc_performance, samples, self.anchor_size)
        max_index, max_improvement = max(enumerate(expected_improvements), key=lambda x: x[1])

        return samples[max_index]

    @staticmethod
    def expected_improvement(model_pipeline: Pipeline, f_star: float, theta: np.array, anchor_size) -> np.array:
        """
        Acquisition function that determines which configurations are good and which
        are not good.

        :param model_pipeline: The internal surrogate model (should be fitted already)
        :param f_star: The current incumbent (theta_inc)
        :param theta: A (n, m) array, each column represents a hyperparameter and each row
        represents a configuration
        :return: A size n vector, same size as each element representing the EI of a given
        configuration
        """
        expected_improvements = []
        tradeoff= 0.1
        for t in theta:
            t_pd = pd.DataFrame([t.get_dictionary()])
            t_pd['anchor_size'] = anchor_size
            mean_x, confidence = model_pipeline.predict(t_pd, return_std=True)

            # Adjust Z calculation with the tradeoff parameter
            Z = (f_star - mean_x - tradeoff) / confidence
            EI = (mean_x - f_star - tradeoff) * norm.cdf(Z) + confidence * norm.pdf(Z)
            
            expected_improvements.append(EI[0])

        return expected_improvements

    def update_runs(self, run: typing.Tuple[typing.Dict, float]):
        """
        After a configuration has been selected and ran, it will be added to the run list
        (so that the model can be trained on it during the next iterations).

        :param run: A tuple (configuration, performance) where performance is error rate
        """
        transformed_run = (run[0], 1 - run[1])
        self.R.append(transformed_run)
        self.fit_model()
        
        run_score = transformed_run[1]
        if run_score > self.theta_inc_performance:
            self.theta_inc = run[0]
            self.theta_inc_performance = run_score





