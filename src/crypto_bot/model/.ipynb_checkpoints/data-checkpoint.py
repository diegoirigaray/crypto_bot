import torch
import numpy as np
import mplfinance as fplt
from torch.utils.data import Dataset, WeightedRandomSampler


PLOT_PARAMS = {
    'type': 'candle',
    'style': 'charles',
    'volume': True,
    'show_nontrading': True
}


def plot_df(df):
    df_plot = df.copy()
    df_plot.set_index('Date', inplace=True, drop=True)
    fplt.plot(df_plot, **PLOT_PARAMS)


def plot_by_date(df, min_date, max_date):
    df_plot = df[(df.Open >= min_date) & (df.Open < max_date)].copy()
    df_plot.set_index('Date', inplace=True, drop=True)
    fplt.plot(df_plot, **PLOT_PARAMS)


def plot_by_idx(df, min_idx, max_idx):
    df_plot = df[(df.index >= min_idx) & (df.index < max_idx)].copy()
    df_plot.set_index('Date', inplace=True, drop=True)
    fplt.plot(df_plot, **PLOT_PARAMS)


def get_class_weights(ds, n_classes):
    class_counts = {}
    binary = len(ds[0][1].size()) == 0
    for c in range(n_classes):
        class_counts[c] = 0
    
    for _, target in ds:
        if binary:
            class_counts[target.item()] += 1
        else:
            class_counts[torch.argmax(target).item()] += 1
    
    weights = []
    for c in range(n_classes):
        weights.append(1 / class_counts[c])
    
    return tuple(weights)


def get_scaled_class_weights(ds, n_classes):
    class_weights = torch.tensor(get_class_weights(ds, n_classes))
    return torch.max(class_weights) / class_weights


def get_weighted_random_sampler(ds, class_weights):
    sample_weights = []
    binary = len(ds[0][1].size()) == 0
    
    for _, target in ds:
        if binary:
            sample_weights.append(class_weights[target.item()])
        else:
            sample_weights.append(class_weights[torch.argmax(target).item()])
    
    sample_weights = torch.from_numpy(np.array(sample_weights))
    return WeightedRandomSampler(sample_weights, len(sample_weights))


class BaseTradingDataset(Dataset):
    REF_COL = 'Close'
    REF_MAX_COL = 'Close'
    REF_MIN_COL = 'Close'

    def __init__(self, df, transforms, train_window_size=25, pred_window_size=15, stride=1):
        self.df = df
        self.transforms = transforms
        self.train_window_size = train_window_size
        self.pred_window_size = pred_window_size
        self.stride = stride

        self._precomputed_data = None
        self._precomputed_target = None

    def __len__(self):
        full_window = self.train_window_size + self.pred_window_size
        return (len(self.df) - full_window + 1) // self.stride
    
    def get_label_from_target(self, price, target):
        pass

    def _get_data(self, idx, raw=False):
        start = idx * self.stride
        end = start + self.train_window_size
        data = self.df[start:end].copy()
        if not raw:
            data = self.transforms.transform(data)
        return data
    
    def _get_target(self, idx, raw=False):
        start = (idx * self.stride) + self.train_window_size
        end = start + self.pred_window_size
        target = self.df[start:end].copy()
        if not raw:
            price = self.df.iloc[start - 1][self.REF_COL]
            target = self.get_label_from_target(price, target)
        return target

    def precompute_data(self):
        data = []
        for i in range(len(self)):
            data.append(self._get_data(i))
        self._precomputed_data = torch.stack(data)
    
    def precompute_target(self):
        target = []
        for i in range(len(self)):
            target.append(self._get_target(i))
        self._precomputed_target = torch.stack(target)

    def precompute(self):
        self.precompute_data()
        self.precompute_target()
    
    def to(self, device):
        if self._precomputed_data is not None:
            self._precomputed_data = self._precomputed_data.to(device)
        if self._precomputed_target is not None:
            self._precomputed_target = self._precomputed_target.to(device)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration
        
        if self._precomputed_data is not None:
            data = self._precomputed_data[idx]
        else:
            data = self._get_data(idx)
        
        if self._precomputed_target is not None:
            target = self._precomputed_target[idx]
        else:
            target = self._get_target(idx)

        # data = torch.tensor(data.values).transpose(0, 1)
        return data, target


class BinaryTradeDataset(BaseTradingDataset):
    def __init__(self, df, transforms, threshold, increase=True, **kwargs):
        super().__init__(df, transforms, **kwargs)
        self.threshold = threshold
        self.increase = increase
    
    def update_target(self, threshold, increase, precompute=True):
        self.threshold = threshold
        self.increase = increase
        
        if precompute:
            self.precompute_target
    
    def get_label_from_target(self, price, target):
        if self.increase:
            max_value = target[self.REF_MAX_COL].max()
            diff = (max_value * 100 / price) - 100
            return torch.tensor((diff > self.threshold) * 1)
        if not self.increase:
            min_value = target[self.REF_MIN_COL].min()
            diff = (min_value * 100 / price) - 100
            return torch.tensor((diff < self.threshold) * 1)


class MulticlassTradeDataset(BaseTradingDataset):
    '''
    Multiclass target with 4 classes:
    Class 1: Upper bound broken without previously breaking lower bound
    Class 2: Lower bound broken without previously breaking upper bound
    Class 3: Else (no bound broken)
    '''
    def __init__(self, df, transforms, sup_threshold, inf_threshold, **kwargs):
        super().__init__(df, transforms, **kwargs)
        self.sup_threshold = sup_threshold
        self.inf_threshold = inf_threshold
    
    def update_target(self, sup_threshold, inf_threshold, precompute=True):
        self.sup_threshold = sup_threshold
        self.inf_threshold = inf_threshold

        if precompute:
            self.precompute_target
    
    def get_label_from_target(self, price, target):
        sup_price = price * (1 + (self.sup_threshold / 100))
        inf_price = price * (1 + (self.inf_threshold / 100))

        over = target[target[self.REF_MAX_COL] > sup_price]
        if not over.empty:
            pos = over.index[0]
            if target[target.index <= pos][self.REF_MIN_COL].min() > inf_price:
                return torch.tensor([1, 0, 0])
        
        if target[self.REF_MIN_COL].min() < inf_price:
            return torch.tensor([0, 1, 0])
            
        return torch.tensor([0, 0, 1])