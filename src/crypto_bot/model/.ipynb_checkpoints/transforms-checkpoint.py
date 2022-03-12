import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class BaseTransform(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class FilterFeatures(BaseTransform):
    '''
    Devuelve un subset del dataframe de entrada unicamente con las
    features seleccionadas.
    '''
    def __init__(self, features=[]):
        self.features = features

    def transform(self, X):
        return X[self.features].copy()


class RelativeTransform(BaseTransform):
    def __init__(self, ref_col, target_cols):
        self.ref_col = ref_col
        self.target_cols = target_cols
    
    def transform(self, X):
        ref_value = X.iloc[-1][self.ref_col]
        ref_factor = 100 / ref_value
        
        X[self.target_cols] = (X[self.target_cols] * ref_factor) - 100
        return X


class ManualNormalizer(BaseTransform):
    def __init__(self, feature, mean, std):
        self.feature = feature
        self.mean = mean
        self.std = std
    
    def transform(self, X):
        X[self.feature] = (X[self.feature] - self.mean) / self.std
        return X


class ToTensorTransform(BaseTransform):
    def transform(self, X):
        return torch.tensor(X.values).transpose(0, 1)