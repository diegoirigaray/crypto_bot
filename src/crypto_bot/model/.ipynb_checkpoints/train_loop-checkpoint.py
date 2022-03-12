import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


class Trainer():
    def __init__(self, model, train_loader, valid_loader=None, epochs=10, lr=0.001, weights=None):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.epochs = epochs
        self.lr = lr
        self.weights = weights
        
        self.train_losses = []
        self.valid_losses = []
        self.prediction_df_train = None
        self.prediction_df_valid = None
        self.binary = len(train_loader.dataset[0][1].size()) == 0

    def setup_device(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Using device: {}'.format(self.device))
        
        self.model.to(self.device)
        self.train_loader.dataset.to(self.device)
        if self.valid_loader is not None:
            self.valid_loader.dataset.to(self.device)
        if self.weights is not None:
            self.weights = self.weights.to(self.device)
    
    def _get_criterion(self):
        if self.binary:
            return nn.BCELoss(self.weights)
        return nn.CrossEntropyLoss(self.weights)
    
    def _get_loss(self, criterion, output, target):
        if self.binary:
            return criterion(output, target.float().unsqueeze(-1))
        return criterion(output, torch.max(target, 1)[1])
    
    def train(self):
        iters = 0
        criterion = self._get_criterion()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        percentage_factor = 100 / len(self.train_loader.dataset)
        self.setup_device()
        
        
        for epoch in range(self.epochs):
            total = 0
            train_loss = 0
            valid_loss = 0
            self.model.train()
            
            for data, target in self.train_loader:
                total += target.size()[0]
                optimizer.zero_grad()
                output = self.model(data.float())
                loss = self._get_loss(criterion, output, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                iters += 1
                if iters % 1000 == 0:
                    print('Epoch: {}/{} - {:.2f}%'.format(epoch + 1, self.epochs, total * percentage_factor))

            log_args = [epoch + 1, self.epochs, (train_loss / len(self.train_loader)), '-']
            if self.valid_loader:
                self.model.eval()
                for data, target in self.valid_loader:
                    valid_loss += self._get_loss(criterion, self.model(data.float()), target).item()
                log_args[-1] = valid_loss / len(self.valid_loader)

            print('Epoch: {}/{} - Train Loss: {} - Valid loss {}'.format(*log_args))

        return self.model
    
    def _get_prediction_df(self, dl):
        if not self.binary:
            raise NotImplementedError
        
        with torch.no_grad():
            _target = []
            _prediction = []
            for data, target in dl:
                output = self.model(data.float())
                _target += target.tolist()
                _prediction += output.flatten().tolist()
            
            return pd.DataFrame({'target': _target, 'pred': _prediction})
    
    def _plot_prediction_distr(self, predictions_df):
        if not self.binary:
            raise NotImplementedError

        predictions_df.groupby('target')['pred'].plot(kind='density', legend=True)
    
    def plot_train_distr(self):
        if self.prediction_df_train is None:
            self.prediction_df_train = self._get_prediction_df(self.train_loader)
        self._plot_prediction_distr(self.prediction_df_train)
    
    def plot_valid_distr(self):
        if self.valid_loader is not None:
            if self.prediction_df_valid is None:
                self.prediction_df_valid = self._get_prediction_df(self.valid_loader)
            self._plot_prediction_distr(self.prediction_df_valid)
    
    def plot_roc(self):
        if not self.binary:
            raise NotImplementedError
        
        if self.prediction_df_train is None:
            self.prediction_df_train = self._get_prediction_df(self.train_loader)
        t_fpr, t_tpr, _ = roc_curve(self.prediction_df_train.target, self.prediction_df_train.pred)
        t_auc = roc_auc_score(self.prediction_df_train.target, self.prediction_df_train.pred)
        plt.plot(t_fpr, t_tpr, label='ROC Train (area = {:3f})'.format(t_auc))
        
        if self.valid_loader is not None:
            if self.prediction_df_valid is None:
                self.prediction_df_valid = self._get_prediction_df(self.valid_loader)
            v_predictions_df = self._get_prediction_df(self.valid_loader)
            v_fpr, v_tpr, _ = roc_curve(self.prediction_df_valid.target, self.prediction_df_valid.pred)
            v_auc = roc_auc_score(self.prediction_df_valid.target, self.prediction_df_valid.pred)
            plt.plot(v_fpr, v_tpr, label='ROC Valid (area = {:3f})'.format(v_auc))
        
        plt.legend(loc=4)
        plt.show()
        return
    
    def find_threshold(self, step=0.025, min_score=0.65):
        if not self.binary or self.valid_loader is None:
            raise NotImplementedError
        if self.prediction_df_valid is None:
                self.prediction_df_valid = self._get_prediction_df(self.valid_loader)

        threshold = 0
        while threshold < 1:
            threshold += step
            segment = self.prediction_df_valid[self.prediction_df_valid.pred > threshold]
            if not len(segment):
                break
            distr = segment.target.value_counts(normalize=True)
            if 1 in distr and distr[1] > min_score:
                return threshold, distr[1]
        return None, None
    
    def print_scores(self, step=0.025, min_threshold=0.3, max_threshold=1):
        if not self.binary or self.valid_loader is None:
            raise NotImplementedError
        if self.prediction_df_valid is None:
                self.prediction_df_valid = self._get_prediction_df(self.valid_loader)
 
        threshold = min_threshold
        while threshold < min(max_threshold, 1):
            segment = self.prediction_df_valid[self.prediction_df_valid.pred > threshold]
            if not len(segment):
                break
            distr = segment.target.value_counts(normalize=True)
            if 1 in distr:
                print('Th: {:.3f} Score: {:.3f} ({})'.format(threshold, distr[1], len(segment)))
            else:
                print('Th: {:.3f} Score:- ({})'.format(threshold, len(segment)))
            threshold += step
