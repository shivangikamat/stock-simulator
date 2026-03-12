import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


ticker=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "DIS","MS"]
data = yf.download(ticker, start="2025-01-01")

data.to_csv("data/stock_data.csv")

# Plotting the price changes for each stock in single plot 
plt.figure(figsize=(14, 7))
for stock in ticker:
    plt.plot(data['Close'][stock], label=stock)     
plt.title('Stock Price Changes from 2025-01-01')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.legend()
plt.grid()
plt.savefig("data/stock_price_changes.png")
plt.show()

# Determmining the percentage change 
pct_change=data['Close'].pct_change().dropna()

# Defining the 20 day momentum
momentum=data['Close'].pct_change(20)

# Creating a long-short strategy based on momentum
def get_positions(signal, n=3):
    # Rank stocks each day
    ranks = signal.rank(axis=1, ascending=False)

    # Long top n
    long = (ranks <= n).astype(int)

    # Short bottom n
    short = (ranks > (len(signal.columns) - n)).astype(int)

    # Combine
    positions = long - short
    return positions

positions=get_positions(momentum)
strategy_returns = (positions.shift(1) * pct_change).mean(axis=1)

sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
annual_return = strategy_returns.mean() * 252

print("Sharpe:", sharpe)
print("Annual Return:", annual_return)

# Monte Carlo Simulation

n_simulations = 1000
n_days = 252

simulated_returns = []
equity_paths = []

historical_returns = strategy_returns.dropna().values

for i in range(n_simulations):

    # randomly sample returns
    sim = np.random.choice(historical_returns, size=n_days, replace=True)

    sim_series = pd.Series(sim)

    equity = (1 + sim_series).cumprod()

    simulated_returns.append(equity.iloc[-1] - 1)

    equity_paths.append(equity)

simulated_returns = np.array(simulated_returns)

print("Monte Carlo Results")
print("Mean Return:", simulated_returns.mean())
print("Median Return:", np.median(simulated_returns))
print("Worst 5%:", np.percentile(simulated_returns, 5))
print("Best 95%:", np.percentile(simulated_returns, 95))

plt.figure(figsize=(12,6))

for i in range(50):
    plt.plot(equity_paths[i], alpha=0.4)

plt.title("Monte Carlo Strategy Simulations")
plt.xlabel("Days")
plt.ylabel("Portfolio Value")
plt.grid()

plt.savefig("data/monte_carlo_paths.png")

plt.show()