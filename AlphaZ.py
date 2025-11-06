"""
This trading algorithm uses mean reversion for alpha generation. We normalise deviation from a rolling average by the local volatility to get a good measure of the deviation. We then sell or buy a single position at thresholds
"""
# Optimised version: kelly
"""
Alpha Generation using Z-scores with Rolling Statistics
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

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
    
    print(f"Downloaded {len(df)} data points for {ticker}")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Latest price: ${df['price'].iloc[-1]:.2f}")
    
    return df


#print("Loading data from Yahoo Finance...")
#tickers = ["AAPL"]

# Download all tickers at once
df = download_stock_data()

# Alpha Generation using Z-scores
def generate_alpha_signals(df, window=20, z_threshold=2.0):
    """
    Generate alpha signals using rolling Z-scores
    Returns DataFrame with signals, z-scores, and alpha values
    """
    df = df.copy()
    
    # Calculate rolling statistics
    df['rolling_mean'] = df['price'].rolling(window=window).mean()
    df['rolling_std'] = df['price'].rolling(window=window).std()
    
    # Calculate Z-score: (current price - rolling mean) / rolling std
    df['z_score'] = (df['price'] - df['rolling_mean']) / df['rolling_std']
    
    # Handle NaN values (first window-1 values)
    df['z_score'].fillna(0, inplace=True)
    
    # Generate signals based on Z-score thresholds
    df['signal'] = 0
    df['signal'] = np.where(df['z_score'] > z_threshold, -1, df['signal'])  # Overbought: SELL
    df['signal'] = np.where(df['z_score'] < -z_threshold, 1, df['signal'])   # Oversold: BUY
    
    # Calculate alpha value (magnitude of deviation from mean)
    df['alpha_value'] = (df['price'] - df['rolling_mean']) / df['price']
    
    return df

# Generate alpha signals
# 23 as window seems optimal
window_size = 23
z_threshold = 1.37  # More sensitive threshold for trading
df_with_alpha = generate_alpha_signals(df, window=window_size, z_threshold=z_threshold)
print(df_with_alpha)

# Backtest the alpha strategy
def backtest_alpha_strategy(df, initial_capital=10000, transaction_cost=0.01):
    """
    Backtest trading strategy based on alpha signals
    """
    capital = initial_capital
    position = 0
    portfolio_values = []
    trades = []
    risk_to_profit_perposition = []
    in_position = False
    buyprice = 0
    
    for i in range(window_size, len(df)):
        current_price = df['price'].iloc[i]
        current_signal = df['signal'].iloc[i]
        current_z = df['z_score'].iloc[i]
        
        # Current portfolio value
        current_value = capital + (position * current_price)
        portfolio_values.append(current_value)
        
        # Trading logic based on Z-score signals
        # BUY SIGNAL
        if current_signal == 1 and not in_position and capital > current_price:
        
            # Position sizing based on kelly with bounds
            if trades == [] or risk_to_profit_perposition == []:
                kelly = 0.2
            else:
                win_rate = len([t for t in trades if t[0] in ['SELL', 'EXIT'] and t[3] > trades[trades.index(t)-1][3]]) / len(trades) if len(trades) > 0 else 0
                risktoprofmean = sum(risk_to_profit_perposition) / len(risk_to_profit_perposition)
                kelly = max(0,min(((risktoprofmean * win_rate) - (1- win_rate))/ risktoprofmean, 0.3))
            
            investment_amount = capital * kelly
            shares_to_buy = investment_amount / current_price
            
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price * (1 + transaction_cost)
                capital -= cost
                position += shares_to_buy
                trades.append(('BUY', df.index[i], shares_to_buy, current_price, current_z))
                buyprice = current_price
                in_position = True
        
        elif current_signal == -1 and in_position and position > 0:  # SELL signal
            # Sell entire position
            proceeds = position * current_price * (1 - transaction_cost)
            capital += proceeds
            trades.append(('SELL', df.index[i], position, current_price, current_z))
            profit = buyprice - current_price
            risk_to_profit_perposition.append(profit/investment_amount)
            position = 0
            in_position = False
        
        # Add mean reversion exit: if position and z-score approaches zero
        elif in_position and abs(current_z) < 0.5 and position > 0:
            proceeds = position * current_price * (1 - transaction_cost)
            capital += proceeds
            trades.append(('EXIT', df.index[i], position, current_price, current_z))
            profit = buyprice - current_price
            position = 0
            in_position = False
    
    # Liquidate final position
    if position > 0:
        proceeds = position * df['price'].iloc[-1] * (1 - transaction_cost)
        capital += proceeds
        trades.append(('SELL', df.index[-1], position, df['price'].iloc[-1], df['z_score'].iloc[-1]))
        position = 0
    
    final_value = capital
    pnl = final_value / initial_capital
    
    return pnl, portfolio_values, trades

# Run backtest
initial_capital = 10000
pnl, portfolio_history, trades = backtest_alpha_strategy(df_with_alpha, initial_capital, transaction_cost=0.01)

# Calculate performance metrics
def calculate_performance_metrics(df, portfolio_history, trades, initial_capital):
    """Calculate various performance metrics"""
    returns = np.diff(portfolio_history) / portfolio_history[:-1]
    
    # Basic metrics
    total_return = (portfolio_history[-1] / initial_capital - 1) * 100
    annualized_return = (1 + total_return/100) ** (252/len(portfolio_history)) - 1
    volatility = np.std(returns) * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    # Drawdown calculation
    peak = np.maximum.accumulate(portfolio_history)
    drawdown = (portfolio_history - peak) / peak
    max_drawdown = np.min(drawdown) * 100
    
    # Trade metrics
    win_rate = len([t for t in trades if t[0] in ['SELL', 'EXIT'] and t[3] > trades[trades.index(t)-1][3]]) / len(trades) if len(trades) > 0 else 0
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'num_trades': len(trades),
        'win_rate': win_rate
    }

# Calculate metrics
metrics = calculate_performance_metrics(df_with_alpha, portfolio_history, trades, initial_capital)

# Visualization
plt.figure(figsize=(15, 12))

# Plot 1: Price and Z-score
plt.subplot(3, 1, 1)
plt.plot(df_with_alpha.index, df_with_alpha['price'], label='Price', linewidth=1)
plt.plot(df_with_alpha.index, df_with_alpha['rolling_mean'], label=f'{window_size}-day MA', alpha=0.7)
plt.fill_between(df_with_alpha.index, 
                df_with_alpha['rolling_mean'] - df_with_alpha['rolling_std'],
                df_with_alpha['rolling_mean'] + df_with_alpha['rolling_std'],
                alpha=0.2, label='±1 STD')
plt.title('Price with Rolling Mean and Standard Deviation')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Z-score and signals
plt.subplot(3, 1, 2)
plt.plot(df_with_alpha.index, df_with_alpha['z_score'], label='Z-score', linewidth=1)
plt.axhline(y=z_threshold, color='r', linestyle='--', alpha=0.7, label=f'Upper threshold ({z_threshold})')
plt.axhline(y=-z_threshold, color='g', linestyle='--', alpha=0.7, label=f'Lower threshold (-{z_threshold})')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)

# Mark buy/sell signals
buy_signals = df_with_alpha[df_with_alpha['signal'] == 1]
sell_signals = df_with_alpha[df_with_alpha['signal'] == -1]
plt.scatter(buy_signals.index, buy_signals['z_score'], color='green', marker='^', s=50, label='Buy Signals')
plt.scatter(sell_signals.index, sell_signals['z_score'], color='red', marker='v', s=50, label='Sell Signals')

plt.title('Z-score with Trading Signals')
plt.ylabel('Z-score')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Portfolio performance
plt.subplot(3, 1, 3)
plt.plot(df_with_alpha.index[window_size:window_size+len(portfolio_history)], 
         portfolio_history, label='Portfolio Value', linewidth=2)
plt.plot(df_with_alpha.index, df_with_alpha['price'] * (initial_capital / df_with_alpha['price'].iloc[window_size]), 
         label='Buy & Hold', alpha=0.7, linewidth=1)
plt.title('Portfolio Performance vs Buy & Hold')
plt.ylabel('Portfolio Value ($)')
plt.xlabel('Date')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print results
print("=" * 60)
print("ALPHA GENERATION USING Z-SCORES - PERFORMANCE RESULTS")
print("=" * 60)
print(f"Initial Capital: ${initial_capital:,.2f}")
print(f"Final Portfolio Value: ${portfolio_history[-1]:,.2f}")
print(f"Total Return: {metrics['total_return']:.2f}%")
print(f"Annualized Return: {metrics['annualized_return']*100:.2f}%")
print(f"Volatility: {metrics['volatility']*100:.2f}%")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
print(f"Number of Trades: {metrics['num_trades']}")
print(f"Win Rate: {metrics['win_rate']*100:.1f}%")
print("=" * 60)

# Show recent signals
print("\nRECENT TRADING SIGNALS:")
recent_signals = df_with_alpha.tail(10)[['price', 'z_score', 'signal', 'alpha_value']]
print(recent_signals)

# Analyze signal effectiveness
print(f"\nSIGNAL ANALYSIS:")
print(f"Buy signals generated: {len(buy_signals)}")
print(f"Sell signals generated: {len(sell_signals)}")
print(f"Total opportunities: {len(buy_signals) + len(sell_signals)}")
