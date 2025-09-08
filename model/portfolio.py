import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from BS import bs_call

#load historical stock data
data = pd.read_csv("data/historical_stock_data.csv", index_col=0, parse_dates=True)
data['Close'] = pd.to_numeric(data['Close'], errors='coerce') #force the string into a number
data = data.dropna(subset=['Close']) #drop rows where 'Close' could not convert

S0 = data['Close'].iloc[-1]  #last closing price
returns = np.log(data['Close'] / data['Close'].shift(1)).dropna() #log returns of closing prices
sigma = returns.std() * np.sqrt(252)
r = 0.05 #assumed risk-free rate

print(f"Spot price: {S0:.2f}, Estimated volatility: {sigma:.4f}")

#define portfolio 
portfolio = [
    {"K": 190, "T": 0.25, "qty": 10},
    {"K": 200, "T": 0.5, "qty": 5},
    {"K": 210, "T": 0.75, "qty": 8},
]

#ccalculate black-scholes prices and portfolio construction
option_names = []
option_values = []

for option in portfolio:
    K = option["K"]
    T = option["T"]
    qty = option["qty"]
    price = bs_call(S0, K, T, r, sigma)
    option_names.append(f"K={K}, T={T}")
    option_values.append(price * qty)

#total portfolio value
total_value = sum(option_values)
print(f"Total portfolio value: ${total_value:.2f}")

#create bar chart
plt.figure(figsize=(10, 6))
plt.bar(option_names, option_values, color='skyblue')
plt.ylabel("Value ($)")
plt.title("Portfolio Option Contributions")
plt.xticks(rotation=45)
plt.grid(axis='y')

#save to results folder
import os
os.makedirs("results", exist_ok=True)

plt.savefig("results/portfolio_chart.png", dpi=300, bbox_inches="tight")
plt.show(block=True)
plt.close()