import ConfigSpace
from sklearn.base import BaseEstimator, TransformerMixin

class ConfigSpaceConverter:
    def __init__(self, config_space):
        self.config_space = config_space
        self.categorical_mappings = self._create_categorical_mappings()

    # create an integer mapping for categorical values
    def _create_categorical_mappings(self):
        mappings = {}
        for hyperparameter in self.config_space.get_hyperparameters():
            if isinstance(hyperparameter, ConfigSpace.hyperparameters.CategoricalHyperparameter):
                mappings[hyperparameter.name] = {choice: idx for idx, choice in enumerate(hyperparameter.choices)}
        return mappings

    # transform the categorical values into integers
    def transform(self, df, complete_defaults= False):
        df_transformed = df.copy()
        if complete_defaults:
            df_transformed= self.complete_with_defaults(df_transformed)
        for column in df_transformed.columns:
            if column in self.categorical_mappings:
                df_transformed[column] = df_transformed[column].map(self.categorical_mappings[column])
        return df_transformed

    def inverse_transform(self, df):
        df_transformed = df.copy()
        for column in df_transformed.columns:
            if column in self.categorical_mappings:
                inverse_mapping = {v: k for k, v in self.categorical_mappings[column].items()}
                df_transformed[column] = df_transformed[column].map(inverse_mapping)
        return df_transformed

    # fill conditional parameters with default values 
    def complete_with_defaults(self, df):
        df_transformed = df.copy()

        # Ensure all hyperparameters are present
        for param in self.config_space.get_hyperparameters():
            if param.name not in df_transformed.columns:
                df_transformed[param.name] = param.default_value
            else:
                df_transformed[param.name] = df_transformed[param.name].fillna(param.default_value)

        # Sort columns to match the order of hyperparameters in the configuration space
        param_order = [param.name for param in self.config_space.get_hyperparameters()]
        param_order.append("anchor_size")
        df_transformed = df_transformed[param_order]  # Reorder the DataFrame columns

        return df_transformed


class ConfigSpaceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, config_space):
        self.config_space = config_space
        self.converter = ConfigSpaceConverter(config_space)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Use transform with complete_defaults=False if called from training

        # During prediction, use complete_defaults=True
        transformed_X= self.converter.transform(X, complete_defaults=True)
        return transformed_X

    def inverse_transform(self, X):
        return self.converter.inverse_transform(X)
    
    
