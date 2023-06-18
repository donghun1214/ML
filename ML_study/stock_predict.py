import requests 

from bs4 import BeautifulSoup

import pandas as pd


investor_df = pd.DataFrame\

(columns=['symbol', 'percent_of_portfolio', 'buys', 'hold_price', 'current_price', 'lowest_price', 'highest_price'])


for i in range(1, 10):

try:

url = 'https://www.dataroma.com/m/g/portfolio_b.php?q=q&o=c&L={}'.format(i)

html = requests.get(url,headers = header)

soup = BeautifulSoup(html.text, 'html.parser')

except:

break 


trs = soup.select('tbody')[0].select('tr')

if len(trs) == 0:

break

for i in range(0, len(trs) - 1, 1):

symbol = trs[i].select('td')[0].text

percent_of_portfolio = trs[i].select('td')[2].text

stock_buys = trs[i].select('td')[3].text

hold_price = trs[i].select('td')[4].text

current_price = trs[i].select('td')[5].text

lowest_price = trs[i].select('td')[6].text

highest_price = trs[i].select('td')[8].text


investor_df = investor_df.append\

({'symbol':symbol, 'percent_of_portfolio':percent_of_portfolio,'buys': stock_buys, 'hold_price': hold_price,'current_price':current_price, 'lowest_price':lowest_price, 'highest_price':highest_price},ignore_index=True)


investor_df.to_parquet('investor.parquet', engine='pyarrow',compression='snappy')


import yfinance as yf import pandas as pd import numpy as np import matplotlib.pyplot as plt import seaborn as sns sns.set_style('whitegrid') plt.style.use("fivethirtyeight") import pandas_datareader.data as datareader import yfinance as yf from datetime import datetimeinvestor_df = pd.read_parquet('investor.parquet') investor_df




