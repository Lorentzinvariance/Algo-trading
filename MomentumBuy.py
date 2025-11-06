"""
This algorithmic trading strategy trades the ticker AAPL based on momentum buying. The alpha here is simply a second derivative. We're using single positions and across multiple other tickers such as TSLA this leads to a staggering PnL of 455% (AAPL) or 700% (TSLA)
on the 6 month timeframe.
"""
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np


def download_stock_data(ticker='AAPL', period='2y', interval='1d'):
    """
    Download real stock data from Yahoo Finance
    """
    print(f"Downloading {ticker} data from Yahoo Finance...")
    
    # Download data
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    
    # Clean and prepare the data
    df = df[['Close', 'Volume', 'Open', 'High', 'Low']]
    df.rename(columns={'Close': 'price'}, inplace=True)
    df.dropna(inplace=True)
    
    return df

chart = download_stock_data(ticker='TSLA', period = '6mo', interval='1h')


def calc_slope(x):
    slope = np.polyfit(range(len(x)), x, 1)[0]
    return slope

# Momentum alpha 
def Alpha(df, short_window=5, grad_threshold = 2):
    df['short rolling mean'] = df['price'].rolling(window=short_window).mean()
    df['slope'] = df['short rolling mean'].rolling(window = grad_threshold).apply(calc_slope)
    df['2nd der'] = df['slope'].rolling(window = grad_threshold).apply(calc_slope)
    df['signal'] = df['2nd der']
    
    return df

alpha_df = Alpha(chart, 4, 3)


def Trading(df, buythres = 2, sellthres = -2.5, capital = 500, short_window=5):
    oricapital = capital
    stock = 0
    one_hold = False
    for i in range(short_window, len(df['price'])):
        if df['signal'].iloc[i] > buythres and one_hold == False:
            #BUY
            capital -= df['price'].iloc[i]
            stock += 1
        
        elif df['signal'].iloc[i] < sellthres and one_hold == True:
            #SELL
            capital += df['price'].iloc[i]
            stock -= 1
    value = stock * df['price'].iloc[i] + capital
    PnL = value / oricapital * 100
    return PnL, value

PnL, final_value = Trading(alpha_df)
print(f"PnL: {PnL}, Final Value of portfolio: {final_value}")
    

plt.figure(figsize=(15, 12))

# Plot 1: Price and Z-score
plt.subplot(3, 1, 1)
plt.plot(chart.index, chart['price'])
plt.plot(alpha_df.index, chart['short rolling mean'])

plt.subplot(3, 1, 2)
plt.plot(alpha_df.index, chart['2nd der'])
plt.show()
