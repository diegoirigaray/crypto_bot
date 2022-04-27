import torch
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from talib.abstract import Function


class BaseTransform(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


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
        gold = {'index': [], 'data': []}
        death = {'index': [], 'data': []}

        for i in range(1, len(X)):
            r1 = X.iloc[i - 1]
            r2 = X.iloc[i]

            if (r1[self.col_a] < r1[self.col_b] and r2[self.col_a] > r2[self.col_b]):
                gold['index'].append(i)
                gold['data'].append(r2[self.col_a])
            elif (r1[self.col_a] > r1[self.col_b] and r2[self.col_a] < r2[self.col_b]):
                death['index'].append(i)
                death['data'].append(r2[self.col_a])

        X['golden' + self.base_name] = pd.Series(**gold)
        X['death' + self.base_name] = pd.Series(**death)
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
        i = self.radius
        result = {'index': [], 'data': []}
        prices = X[self.CLOSE]
        
        while i < (len(X) - self.radius):
            curr = prices.iloc[i]
            limit = getattr(prices.iloc[(i - self.radius):(i + self.radius + 1)], self.FUNC)()
            
            if curr == limit:
                result['index'].append(i)
                result['data'].append(curr)
            
            i += 1 if (curr != limit) else (self.radius + 1)
        
        X[self.result_name] = pd.Series(**result)
        return X


class LocalMinTransform(LocalMinMaxBaseTransform):
    FUNC = 'min'


class LocalMaxTransform(LocalMinMaxBaseTransform):
    FUNC = 'max'


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
    def __init__(self, steps, values_col, result_name, drop_last=0, run_incomplete=False):
        self.steps = sorted(steps if isinstance(steps, list) else [steps])
        self.values_col = values_col
        self.result_name = result_name
        self.drop_last = drop_last
        self.run_incomplete = run_incomplete
    
    def _get_objective(self, subset):
        pass
    
    def _get_result_name(self, steps):
        return '{}{}'.format(self.result_name, steps)
    
    def transform(self, X):
        values = X[self.values_col]
        results = [X]
        
        for s in self.steps:
            result = {'index': [], 'data': [], 'name': self._get_result_name(s)}
            start = 0 if self.run_incomplete else (s - 1)
            
            for i in range(start, len(X)):
                subset = values[max(0, (i + 1 - s)):(i + 1 - self.drop_last)]
                res_i = self._get_objective(subset)
                
                if res_i is not None:
                    result['index'].append(i)
                    result['data'].append(res_i)
            results.append(pd.Series(**result, dtype='object'))
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
        super().__init__(steps, values_col, result_name, drop_last=radius, run_incomplete=False)
    
    def _sorted(self, values):
        for i in range(1, len(values)):
            if self.FUNC == 'higher':
                if values[i] < values[i - 1]:
                    return False
            else:
                if values[i] > values[i - 1]:
                    return False
        return True
    
    def _get_objective(self, subset):
        subset = subset.dropna().tolist()
        return len(subset) if self._sorted(subset) else None


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
        result_name = '{}{}'.format(self.FUNC, suffix)
        super().__init__(steps, values_col, result_name, drop_last=radius, run_incomplete=False)

    def _get_objective(self, subset):
        val = getattr(subset, self.FUNC)()
        return val if val else None


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
        super().__init__(steps, values_col, result_name, drop_last=drop_last, run_incomplete=False)

    def _get_objective(self, subset):
        if self.cond == 'eq':
            count = (subset == self.value).sum()
        elif self.cond == 'gt':
            count = (subset > self.value).sum()
        elif self.cond == 'lt':
            count = (subset < self.value).sum()
        elif self.cond == 'gte':
            count = (subset >= self.value).sum()
        elif self.cond == 'lte':
            count = (subset <= self.value).sum()
        elif self.cond == 'neq':
            count = (subset != self.value).sum()
        return count


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

    def __init__(self, base_col, radius=2, bounce_strength=1):
        self.base_col = base_col
        self.radius = radius
        self.bounce_strength = bounce_strength
        self.result_name = '{}{}_{}'.format('sup' if self.TYPE == 'support' else 'res', radius, base_col)
    
    def transform(self, X):
        i = self.radius
        indexes = []
        values = []
        closes = X[self.CLOSE]
        extremes = X[self.HIGH] if self.TYPE == 'resistance' else X[self.LOW]
        base = X[self.base_col]
        
        while i < len(X) - self.radius:
            m, c_l, c_r = extremes.iloc[i], closes.iloc[i - self.radius], closes.iloc[i + self.radius]
            b_m, b_l, b_r = base.iloc[i], base.iloc[i - self.radius], base.iloc[i + self.radius]
            sup = ((self.TYPE == 'support') and (m < b_m) and (c_l > b_l) and (c_r > b_r))
            res = ((self.TYPE != 'support') and (m > b_m) and (c_l < b_l) and (c_r < b_r))
            # dist_cond = (
            #     (self.bounce_strength * abs(m - b_m) < abs(c_l - b_l)) and
            #     (self.bounce_strength * abs(m - b_m) < abs(c_r - b_r))
            # )
            dist_cond = True
            local_end = m == getattr(
                extremes.iloc[(i - self.radius):(i + self.radius + 1)],
                'min' if self.TYPE == 'support' else 'max'
            )()
            if (sup or res) and dist_cond and local_end:
                indexes.append(i)
                values.append(b_m)
                i += self.radius + 1
            else:
                i += 1
        
        X[self.result_name] = pd.Series(values, index=indexes)
        return X


class SupportBounceTransform(SupportResistanceBounceBaseTransform):
    TYPE = 'support'


class ResistanceBounceTransform(SupportResistanceBounceBaseTransform):
    TYPE = 'resistance'


class RowRelativeTransform(BaseCandlestickTransform):
    def __init__(self, target_cols, ref_col=None):
        self.target_cols = target_cols
        self.ref_col = ref_col if ref_col else self.CLOSE
    
    def transform(self, X):
        ref_factors = 100 / X[self.ref_col]
        X[self.target_cols] = (X[self.target_cols].multiply(ref_factors, axis="index")) - 100
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
        return pd.DataFrame({f: X.get(f, pd.Series(index=X.index, name=f)) for f in self.features})


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
        X.drop(self.drop, axis=1, inplace=True)
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

