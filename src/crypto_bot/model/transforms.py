import torch
import pandas as pd
import numpy as np
from pandas.core.indexers.objects import BaseIndexer, FixedForwardWindowIndexer
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from talib.abstract import Function


class BaseTransform(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class PrintTransform(BaseTransform):
    def __init__(self, text):
        self.text = text

    def fit(self, X, y=None):
        print('{} Fit -> {}'.format(datetime.now().strftime("%H:%M:%S"), self.text))
        return super().fit(X, y)
    
    def transform(self, X):
        print('{} Transform -> {}'.format(datetime.now().strftime("%H:%M:%S"), self.text))
        return super().transform(X)


class BaseCandlestickTransform(BaseTransform):
    OPEN = 'Open'
    HIGH = 'High'
    LOW = 'Low'
    CLOSE = 'Close'
    VOLUME = 'Volume'


# Pytorch model transforms
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


class Normalizer(BaseTransform):
    def __init__(self, features, exclude=[]):
        self.features = features
        self.exclude = exclude
        self.data = {}
    
    def fit(self, X, y=None):
        features = X.columns.tolist() if self.features == 'all' else self.features
        features = [f for f in features if f not in self.exclude]
        for f in features:
            self.data[f] = {'mean': X[f].mean(), 'std': X[f].std()}
        return self
    
    def transform(self, X):
        for f, data in self.data.items():
            X[f] = (X[f] - data['mean']) / data['std']
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


# Sickit learn model transforms
class TalibBaseTransform(BaseCandlestickTransform):
    '''
    Base Class for using talib functions as transforms
    https://github.com/mrjbq7/ta-lib
    '''
    def __init__(self, func_name, output_name, **kwargs):
        self.func_name = func_name
        self.output_name = output_name
        self.kwargs = kwargs
        self.f = Function(func_name)
    
    def _get_input(self, X):
        inputs = {
            'open': X[self.OPEN],
            'high': X[self.HIGH],
            'low': X[self.LOW],
            'close': X[self.CLOSE],
            'volume': X[self.VOLUME]
        }
        return inputs
    
    def transform(self, X):
        inputs = self._get_input(X)
        output = self.f(inputs, **self.kwargs)
        if len(output) == len(X):
            X[self.output_name] = output
        else:
            names = {k: '{}{}'.format(self.output_name, k) for k in range(len(output))}
            output = pd.DataFrame(output).transpose().rename(columns=names)
            X = pd.concat([X, output], axis=1)
        return X


class MultiPatternTransform(TalibBaseTransform):
    '''
    Transform for adding multiple talib candlestick patterns.
    '''
    def __init__(self, func_names):
        self.func_names = func_names if isinstance(func_names, list) else [func_names]
        self.fs = {func.lower(): Function(func) for func in self.func_names}
    
    def transform(self, X):
        inputs = self._get_input(X)
        for name, f in self.fs.items():
            X[name] = f(inputs)
        return X


class MovingAverageTransform(TalibBaseTransform):
    '''
    DEMA                 Double Exponential Moving Average
    EMA                  Exponential Moving Average
    KAMA                 Kaufman Adaptive Moving Average
    MA                   Moving average
    MAMA                 MESA Adaptive Moving Average
    MAVP                 Moving average with variable period
    SMA                  Simple Moving Average
    T3                   Triple Exponential Moving Average (T3)
    TEMA                 Triple Exponential Moving Average
    TRIMA                Triangular Moving Average
    WMA                  Weighted Moving Average
    '''
    def __init__(self, func_name, timeperiod=30, **kwargs):
        self.func_name = func_name
        self.timeperiod = timeperiod if isinstance(timeperiod, list) else [timeperiod]
        self.output_name = func_name.lower()
        super().__init__(func_name, self.output_name, timeperiod=timeperiod, **kwargs)
    
    def transform(self, X):
        inputs = self._get_input(X)
        for tp in self.timeperiod:
            X['{}{}'.format(self.output_name, tp)] = self.f(inputs, timeperiod=tp)
        return X


class GoldenDeathCrossTransform(BaseCandlestickTransform):
    '''
    Calculates golden cross and death cross for the specified columns.
    '''
    def __init__(self, col_a, col_b):
        self.col_a = col_a
        self.col_b = col_b
        self.base_name = '_{}_{}'.format(col_a, col_b)
    
    def transform(self, X):
        diff = X[self.col_a] - X[self.col_b]
        shifted = diff.shift(1, fill_value=0)
        
        gold = ((shifted < 0) & (diff > 0))
        death = ((shifted > 0) & (diff < 0))
        
        X.loc[gold, 'golden' + self.base_name] = X[self.col_a]
        X.loc[death, 'death' + self.base_name] = X[self.col_a]
        return X


class LocalMinMaxBaseTransform(BaseCandlestickTransform):
    '''
    Creates a column with the close values that are local min/max
    in the specified radius.
    '''
    FUNC = None

    def __init__(self, radius=3):
        self.radius = radius
        self.result_name = 'local{}{}'.format(self.FUNC, radius)
    
    def transform(self, X):
        limit = getattr(X[self.CLOSE].rolling(window=self.radius * 2, min_periods=0, center=True), self.FUNC)()
        cond = limit == X[self.CLOSE]
        X.loc[cond, self.result_name] = X[self.CLOSE]
        return X


class LocalMinTransform(LocalMinMaxBaseTransform):
    FUNC = 'min'


class LocalMaxTransform(LocalMinMaxBaseTransform):
    FUNC = 'max'


class FixedLastDropedWindowIndexer(BaseIndexer):
    def __init__(self, index_array=None, window_size=0, drop_last=0):
        self.index_array = index_array
        self.window_size = window_size
        self.drop_last = drop_last

    def get_window_bounds(self, num_values=0, min_periods=None, center=None, closed=None):
        if center:
            offset = (self.window_size - 1) // 2
        else:
            offset = 0

        end = np.arange(1 + offset, num_values + 1 + offset, dtype="int64")
        start = end - self.window_size
        if self.drop_last:
            end -= self.drop_last
        if closed in ["left", "both"]:
            start -= 1
        if closed in ["left", "neither"]:
            end -= 1

        end = np.clip(end, 0, num_values)
        start = np.clip(start, 0, num_values)

        return start, end


class MovingWindowBaseTransform(BaseCandlestickTransform):
    '''
    Base transform for features based on a moving window over
    some specified column.
    Given one or many window sizes, creates an new feature for each
    one of them, where the calculated value corresponds to the last
    element of the moving window.
    
    *drop_last is usefull for cases when the last n elements of the window
    arent expected to be available in execution time.
    '''
    def __init__(self, steps, values_col, result_name, drop_last=0):
        self.steps = sorted(steps if isinstance(steps, list) else [steps])
        self.values_col = values_col
        self.result_name = result_name
        self.drop_last = drop_last

    def _get_result_name(self, steps):
        return '{}{}'.format(self.result_name, steps)
    
    def _get_values(self, X):
        return X[self.values_col]
    
    def _get_window(self, window_size):
        if not self.drop_last:
            return window_size
        return FixedLastDropedWindowIndexer(window_size=window_size, drop_last=self.drop_last)

    def _get_objective(self, roll):
        pass
    
    def transform(self, X):
        results = [X]

        for s in self.steps:
            values = self._get_values(X).copy()
            window = self._get_window(s)
            roll = values.rolling(window=window, min_periods=0)
            result = self._get_objective(roll)
            result.name = self._get_result_name(s)
            results.append(result)
        
        X = pd.concat(results, axis=1)
        return X


class HighersLowersBaseTransform(MovingWindowBaseTransform):
    '''
    Transform based on moving windows that checks if the 
    values in the specified column are all ascending/descending
    in the current window.
    '''
    FUNC = None

    def __init__(self, steps, values_col, suffix, radius=2):
        result_name = '{}{}_'.format(self.FUNC, suffix)
        self.monotonic ='is_monotonic' if self.FUNC == 'higher' else 'is_monotonic_decreasing'
        super().__init__(steps, values_col, result_name, drop_last=radius)
    
    def _get_objective_item(self, subset):
        non_empty = subset.dropna()
        is_sorted = getattr(non_empty, self.monotonic)
        return len(non_empty) if is_sorted else 0
    
    def _get_objective(self, roll):
        return roll.apply(self._get_objective_item)


class HighersTransform(HighersLowersBaseTransform):
    FUNC = 'higher'


class LowersTransform(HighersLowersBaseTransform):
    FUNC = 'lower'


class MinMaxBaseTransform(MovingWindowBaseTransform):
    '''
    Transform based on moving window that gets the min/max
    value for the given column in the current window.
    '''
    FUNC = None
    
    def __init__(self, steps, values_col, suffix, radius=2):
        result_name = '{}_{}'.format(self.FUNC, suffix)
        super().__init__(steps, values_col, result_name, drop_last=radius)
    
    def _get_objective(self, roll):
        return getattr(roll, self.FUNC)()


class MinWindowTransform(MinMaxBaseTransform):
    FUNC = 'min'


class MaxWindowTransform(MinMaxBaseTransform):
    FUNC = 'max'


class CountCondTransform(MovingWindowBaseTransform):
    '''
    Transform based on moving window that counts the total
    of elements of the specifiad column that match the given condition
    inside the current window.
    '''
    def __init__(self, steps, values_col, result_name, value, cond='eq', drop_last=0):
        self.value = value
        self.cond = cond
        super().__init__(steps, values_col, result_name, drop_last=drop_last)

    def _get_values(self, X):
        ds = X[self.values_col]
        if self.cond == 'eq':
            cond = (ds == self.value)
        elif self.cond == 'gt':
            cond = (ds > self.value)
        elif self.cond == 'lt':
            cond = (ds < self.value)
        elif self.cond == 'gte':
            cond = (ds >= self.value)
        elif self.cond == 'lte':
            cond = (ds <= self.value)
        elif self.cond == 'neq':
            cond = (ds != self.value)
        return cond
    
    def _get_objective(self, roll):
        return roll.sum()


class CountCondCrossTransform(MovingWindowBaseTransform):
    '''
    TBD
    '''
    pass


class SupportResistanceBounceBaseTransform(BaseCandlestickTransform):
    '''
    Transform to spot when close prices bounce on a support/resistance
    as given by the specified column value.
    '''
    TYPE = None

    def __init__(self, base_col, radius=2):
        self.base_col = base_col
        self.radius = radius
        self.result_name = '{}{}_{}'.format('sup' if self.TYPE == 'support' else 'res', radius, base_col)

    def transform(self, X):
        extreme = X[self.HIGH] if self.TYPE == 'resistance' else X[self.LOW]
        extreme_diff = extreme - X[self.base_col]
        close_diff = X[self.CLOSE] - X[self.base_col]
        left_shift = close_diff.shift(self.radius, fill_value=0)
        right_shift = close_diff.shift(-self.radius, fill_value=0)
        
        roll = extreme.rolling(window=self.radius, min_periods=0, center=True)
        local_extreme = (roll.min() == extreme) if self.TYPE == 'support' else (roll.max() == extreme)
        
        if self.TYPE == 'support':
            bounce_cond = (left_shift > 0) & (right_shift > 0) & (extreme_diff < 0) & local_extreme
        else:
            bounce_cond = (left_shift < 0) & (right_shift < 0) & (extreme_diff > 0) & local_extreme
        
        X.loc[bounce_cond, self.result_name] = X[self.base_col]
        return X


class SupportBounceTransform(SupportResistanceBounceBaseTransform):
    TYPE = 'support'


class ResistanceBounceTransform(SupportResistanceBounceBaseTransform):
    TYPE = 'resistance'


class RowRelativeTransform(BaseCandlestickTransform):
    def __init__(self, target_cols, ref_col=None, variation=True):
        self.target_cols = target_cols
        self.ref_col = ref_col if ref_col else self.CLOSE
        self.variation = variation
    
    def transform(self, X):
        ref_factors = 100 / X[self.ref_col]
        X[self.target_cols] = (X[self.target_cols].multiply(ref_factors, axis="index"))
        if self.variation:
            X[self.target_cols] -= 100
        return X


class FilterFeatures(BaseTransform):
    '''
    Devuelve un subset del dataframe de entrada unicamente con las
    features seleccionadas.
    Puede agregar columnas vacías si se seleccionan features
    que no estaban en el dataframe original.
    '''
    def __init__(self, features=[]):
        self.features = features
    
    def transform(self, X):
        data = {f: X.get(f, pd.Series(index=X.index, name=f, dtype=object)) for f in self.features}
        return pd.DataFrame(data)


class DropFeatures(BaseTransform):
    '''
    Elimina las features especificadas.
    Opcionalmente se puede elegir eliminar las features que unicamente
    presentan nulls o un valor constante en el dataframe de entrenamiento.
    '''
    def __init__(self, features=[], drop_constant=True):
        self.features = features
        self.drop_constant = drop_constant

    def fit(self, X, y=None):
        self.keep = []
        self.const = []
        self.drop = [f for f in self.features if f in X.columns]

        if self.drop_constant:
            self.const = X.columns[X.nunique(dropna=False) == 1].tolist()
            self.drop = list(set(self.drop + self.const))
        else:
            self.drop = list(set(self.drop))

        self.keep = [f for f in X.columns if f not in self.drop]
        return self

    def transform(self, X):
        if hasattr(self, 'drop'):
            drop = self.drop
        else:
            drop = self.features
        X.drop(drop, axis=1, inplace=True)
        return X


class NullImputer(BaseTransform):
    def __init__(self, make_copy=False):
        self.make_copy = make_copy

    def transform(self, X):
        if self.make_copy:
            X = X.copy()
        X.fillna(np.nan, inplace=True)
        return X

    
class FeaturesImputer(BaseTransform):
    '''
    Aplica un `SimpleImputer` unicamente a un subconjunto
    de features.
    '''
    def __init__(self, features=[], **kwargs):
        self.features = features
        self.imputer = SimpleImputer(**kwargs)

    def fit(self, X, y=None):
        if self.features == 'all':
            self.features_ = X.columns.tolist()
        else:
            self.features_ = [f for f in self.features if f in X.columns]

        if self.features_:
            sub_X = X[self.features_]
            self.dtypes = sub_X.dtypes
            self.imputer.fit(sub_X, y)
        return self

    def transform(self, X):
        if self.features_:
            sub_X = X[self.features_].astype(self.dtypes)
            res = pd.DataFrame(
                self.imputer.transform(sub_X),
                columns=self.features_, index=X.index
            )
            X[self.features_] = res
        return X


class MultiFeaturesImputer(BaseTransform):
    '''
    Aplica multiples `FeaturesImputer` en un mismo transform.
    '''
    def __init__(self, config=[]):
        self.config = config

    def fit(self, X, y=None):
        self.imputers = [FeaturesImputer(**conf) for conf in self.config]
        for imputer in self.imputers:
            imputer.fit(X, y)
        return self

    def transform(self, X):
        for imputer in self.imputers:
            X = imputer.transform(X)
        return X


class ToFloatTransform(BaseTransform):
    def __init__(self, columns='all'):
        self.columns = columns

    def transform(self, X):
        if self.columns == 'all':
            return X.astype('float')
        for col in self.columns:
            X[col] = X[col].astype('float')
        return X


class RegisterFeatures(BaseTransform):
    def __init__(self, to_numpy=False):
        self.to_numpy = to_numpy
        self.features = None

    def fit(self, X, y=None):
        self.features = X.columns.values.tolist()
        return self
    
    def transform(self, X):
        if self.to_numpy:
            return X.values
        return X


class RiseTargetTransform(MovingWindowBaseTransform):
    def __init__(self, steps, above, not_below, result_name):
        self.above = above
        self.not_below = not_below
        super().__init__(steps, self.CLOSE, result_name, drop_last=0)
    
    def _get_window(self, window_size):
        return FixedForwardWindowIndexer(window_size=window_size)
    
    def _get_objective_item(self, subset):
        above_cond = True
        not_below_cond = True
        
        current_ratio = 100 / subset.iloc[0]
        max_var = (subset.max() * current_ratio) - 100
        min_var = (subset.min() * current_ratio) - 100
        
        if self.above is not None:
            above_cond = max_var > self.above
        if self.not_below is not None:
            not_below_cond = min_var > self.not_below
        
        return int(above_cond and not_below_cond)
    
    def _get_objective(self, roll):
        return roll.apply(self._get_objective_item)