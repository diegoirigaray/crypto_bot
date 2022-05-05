# Simplification of: https://github.com/binance/binance-public-data/tree/master/python

import pandas as pd
from datetime import *
from dateutil.rrule import rrule, MONTHLY

import os, sys
from pathlib import Path
import urllib.request


YEARS = ['2017', '2018', '2019', '2020', '2021', '2022']
INTERVALS = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1mo"]
DAILY_INTERVALS = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]
TRADING_TYPE = ["spot", "um", "cm"]
MONTHS = list(range(1,13))
MAX_DAYS = 35
BASE_URL = 'https://data.binance.vision/'
START_DATE = date(int(YEARS[0]), MONTHS[0], 1)
END_DATE = datetime.date(datetime.now())


def get_destination_dir(file_url, folder=None):
  store_directory = os.environ.get('STORE_DIRECTORY')
  if folder:
    store_directory = folder
  if not store_directory:
    store_directory = os.path.dirname(os.path.realpath(__file__))
  return os.path.join(store_directory, file_url)


def get_download_url(file_url):
  return "{}{}".format(BASE_URL, file_url)


def download_file(base_path, file_name, date_range=None, folder=None):
  download_path = "{}{}".format(base_path, file_name)
  if folder:
    base_path = os.path.join(folder, base_path)
  if date_range:
    date_range = date_range.replace(" ","_")
    base_path = os.path.join(base_path, date_range)
  save_path = get_destination_dir(os.path.join(base_path, file_name), folder)
  

  if os.path.exists(save_path):
    print("\nfile already exists! {}".format(save_path))
    return
  
  # make the directory
  if not os.path.exists(base_path):
    Path(get_destination_dir(base_path)).mkdir(parents=True, exist_ok=True)

  try:
    download_url = get_download_url(download_path)
    dl_file = urllib.request.urlopen(download_url)
    length = dl_file.getheader('content-length')
    if length:
      length = int(length)
      blocksize = max(4096,length//100)

    with open(save_path, 'wb') as out_file:
      dl_progress = 0
      print("\nFile Download: {}".format(save_path))
      while True:
        buf = dl_file.read(blocksize)   
        if not buf:
          break
        dl_progress += len(buf)
        out_file.write(buf)
        done = int(50 * dl_progress / length)
        sys.stdout.write("\r[%s%s]" % ('#' * done, '.' * (50-done)) )    
        sys.stdout.flush()

  except urllib.error.HTTPError:
    print("\nFile not found: {}".format(download_url))
    pass

def convert_to_date_object(d):
  year, month, day = [int(x) for x in d.split('-')]
  date_obj = date(year, month, day)
  return date_obj

def get_path(trading_type, market_data_type, time_period, symbol, interval=None):
  trading_type_path = 'data/spot'
  if trading_type != 'spot':
    trading_type_path = f'data/futures/{trading_type}'
  if interval is not None:
    path = f'{trading_type_path}/{time_period}/{market_data_type}/{symbol.upper()}/{interval}/'
  else:
    path = f'{trading_type_path}/{time_period}/{market_data_type}/{symbol.upper()}/'
  return path


def download_monthly_klines(trading_type, symbols, num_symbols, intervals, years, months, start_date, end_date, folder, checksum):
  current = 0
  date_range = None

  # if start_date and end_date:
  #   date_range = start_date + " " + end_date

  if not start_date:
    start_date = START_DATE
  else:
    start_date = convert_to_date_object(start_date)

  if not end_date:
    end_date = END_DATE
  else:
    end_date = convert_to_date_object(end_date)

  print("Found {} symbols".format(num_symbols))

  for symbol in symbols:
    print("[{}/{}] - start download monthly {} klines ".format(current+1, num_symbols, symbol))
    for interval in intervals:
      for year in years:
        for month in months:
          current_date = convert_to_date_object('{}-{}-01'.format(year, month))
          if current_date >= start_date and current_date <= end_date:
            path = get_path(trading_type, "klines", "monthly", symbol, interval)
            file_name = "{}-{}-{}-{}.zip".format(symbol.upper(), interval, year, '{:02d}'.format(month))
            download_file(path, file_name, date_range, folder)

            if checksum == 1:
              checksum_path = get_path(trading_type, "klines", "monthly", symbol, interval)
              checksum_file_name = "{}-{}-{}-{}.zip.CHECKSUM".format(symbol.upper(), interval, year, '{:02d}'.format(month))
              download_file(checksum_path, checksum_file_name, date_range, folder)

    current += 1


# CUSTOM CODE
def download_data(folder, type='spot', symbols=['BTCUSDT'], intervals=['1m'], years=YEARS, months=MONTHS, startDate=None, endDate=None, checksum=0):
    num_symbols = len(symbols)
    download_monthly_klines(type, symbols, num_symbols, intervals, years, months, startDate, endDate, folder, checksum)


def load_dataframe(folder, start_date, end_date, type='spot', symbol='BTCUSDT', interval='1m'):
  cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'close_date', 'quote_asset_volume', 'number_of_trades',	'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',	'ignore']
  path = '{}/data/{}/monthly/klines/{}/{}/'.format(folder, type, symbol, interval)

  data = []
  start, end = datetime.strptime(start_date, '%Y-%m-%d'), datetime.strptime(end_date, '%Y-%m-%d')
  rr = rrule(MONTHLY, dtstart=start, until=end)
  for p in rr:
    file_name = '{}-{}-{}-{:02d}.zip'.format(symbol, interval, p.year, p.month)
    try:
        data.append(pd.read_csv(path + file_name, names=cols))
    except FileNotFoundError:
        print('File "{}" not found'.format(file_name))
    except Exception:
        print('Ouch!')
  
  data = pd.concat(data, ignore_index=True)
  data['Date'] = pd.to_datetime(data['Date'], unit='ms')
  data['close_date'] = pd.to_datetime(data['close_date'], unit='ms')
  return data
