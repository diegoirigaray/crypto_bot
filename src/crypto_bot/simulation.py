import pandas as pd


# Model wrappers
class BaseTradingModel():
    def __init__(self, *args, **kwargs):
        pass
    
    def predict(self, data):
        pass


class TorchTradingNet(BaseTradingModel):
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def predict(self, data):
        return self.model(data.unsqueeze(0).float())[0].item()


class SickitLearnTradingModel(BaseTradingModel):
    def __init__(self, model):
        self.model = model

    def predict(self, data):
        _data = pd.DataFrame([data])
        return self.model.predict_proba(_data).flatten()[1]


# Traders
class BaseTrader():
    def should_buy(self, *args, **kwargs):
        pass
    
    def should_sell(self, *args, **kwargs):
        pass
    
    def buy(self, price):
        pass
    
    def sell(self):
        pass


class BuyModelLossLimitTrader(BaseTrader):
    def __init__(self, model, buy_threshold, hold_threshold, target_profit=0.7, loss_limit=0.3):
        self.model = model
        self.buy_threshold = buy_threshold
        self.hold_threshold = hold_threshold
        self.target_profit = 100 + target_profit
        self.opt_target_profit = 100 + (target_profit * 1.5)
        self.loss_limit = 100 - loss_limit
        
        self.bought_price = None
        self.max_owned_price = 0
    
    def should_buy(self, data, *args, **kwargs):
        score = self.model.predict(data)
        if score >= self.buy_threshold:
            return True
        return False
    
    def buy(self, price):
        self.bought_price = price
        self.max_owned_price = price
    
    def sell(self, *args, **kwargs):
        self.bought_price = None
        self.max_owned_price = 0
    
    def should_sell(self, data, price):
        perc = price * 100 / self.bought_price
        self.max_owned_price = max(self.max_owned_price, price)

        # Limit loss
        if perc < self.loss_limit:
            return True

        decreasing = price < self.max_owned_price
        should_hold = self.model.predict(data) > self.hold_threshold

        # Already proffitable and not going up
        proffitable = perc > self.target_profit
        if proffitable and decreasing and not should_hold:
            return True

        # Too proffitable
        too_proffitable = perc > self.opt_target_profit
        if proffitable and decreasing and not should_hold:
            return True

        return False


class Simulator():
    def __init__(self, ds, balance, trader, df=True, sup_df=None):
        self.ds = ds
        self.initial_balance = balance
        self.dol_balance = balance
        self.crypto_balance = 0
        self.trader = trader
        self.df = df
        self.sup_df = sup_df
    
    def _get_data_info(self, ds, index):
        if self.df:
            data = ds.iloc[index]
            current_date = data.Date
            current_price = data.Close
        else:
            data, _ = ds[index]
            if self.sup_df is None:
                raw_data = ds._get_data(index, raw=True)
                current_date = raw_data.iloc[-1].Date
                current_price = raw_data.iloc[-1].Close
            else:
                pos = (index * ds.stride) + ds.window_size - 1
                current_date = self.sup_df.iloc[pos].Date
                current_price = self.sup_df.iloc[pos].Close
        return current_date, current_price, data

    def simulate(self):
        buy_orders = {'data': [], 'index': []}
        sell_orders = {'data': [], 'index': []}
        
        for i in range(len(self.ds)):
            current_date, current_price, data = self._get_data_info(self.ds, i)
            
            if self.dol_balance:
                should_buy = self.trader.should_buy(data)
                if should_buy:
                    self.trader.buy(current_price)
                    self.crypto_balance = (self.dol_balance / current_price) * 0.99925
                    spent = self.dol_balance
                    self.dol_balance = 0
                    
                    buy_orders['index'].append(i)
                    buy_orders['data'].append(spent)
                    print('{} Price: {} Bought: {} (${})'.format(
                        current_date,
                        current_price,
                        self.crypto_balance,
                        spent
                    ))
            
            if self.crypto_balance:
                should_sell = self.trader.should_sell(data, current_price)
                if should_sell:
                    self.trader.sell()
                    sold = self.crypto_balance
                    self.dol_balance = (self.crypto_balance * current_price) * 0.99925
                    self.crypto_balance = 0
                    
                    sell_orders['index'].append(i)
                    sell_orders['data'].append(self.dol_balance)
                    print('{} Price: {} Sold: {} (${})'.format(
                        current_date,
                        current_price,
                        sold,
                        self.dol_balance
                    ))
        if self.sup_df is None:
            self.ds['buy_orders'] = pd.Series(**buy_orders)
            self.ds['sell_orders'] = pd.Series(**sell_orders)
        else:
            self.sup_df['buy_orders'] = pd.Series(**buy_orders)
            self.sup_df['sell_orders'] = pd.Series(**sell_orders)