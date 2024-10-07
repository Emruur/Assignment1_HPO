import ConfigSpace

import sklearn.impute
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import pandas as pd

from ConfigConverter import ConfigSpaceConverter


class SurrogateModel:

    def __init__(self, config_space):
        self.config_space = config_space
        self.df = None
        self.model = None
        self.config_converter= ConfigSpaceConverter(config_space)

    def fit(self, df):
        """
        Receives a DataFrame where each column (except the last two) represents a hyperparameter,
        the penultimate column is the anchor size, and the final column is the performance score.

        :param df: DataFrame containing hyperparameters and performance score.
        :return: None. Trains a RandomForestRegressor model and stores it in self.model.
        """

        # Transform the DataFrame with configuration settings
        df = self.config_converter.transform(df)
        # Define feature matrix X and target vector y
        X = df.drop('score', axis=1)
        y = df['score']

        # Optional: Split data for training and testing

        # Initialize RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=100, random_state=42)

        # Train the model on the training data
        rf.fit(X, y)

        # Store the trained model
        self.model = rf

    def predict(self, theta_new):
        """
        Predicts the performance of a given configuration theta_new

        :param theta_new: a dict, where each key represents the hyperparameter (or anchor)
        :return: float, the predicted performance of theta new (which can be considered the ground truth)
        """
        # Convert the dictionary to a DataFrame
        df_input = pd.DataFrame([theta_new])
        df_input= self.config_converter.transform(df_input, complete_defaults=True)

        # Make predictions using the trained model
        predictions = self.model.predict(df_input)
        return predictions[0]
