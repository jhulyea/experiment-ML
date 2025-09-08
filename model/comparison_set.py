import numpy as np
import pandas as pd
from BS import bs_call 
from ml_model import model
import matplotlib.pyplot as plt

#load historical stock data
data = pd.read_csv("data/historical_stock_data.csv", index_col=0, parse_dates=True)

data['Close'] = pd.to_numeric(data['Close'], errors='coerce') #force the string into a number
data = data.dropna(subset=['Close']) #drop rows where 'Close' could not be converted

S0 = data['Close'].iloc[-1]  #last closing price

#volatility
returns = np.log(data['Close'] / data['Close'].shift(1)).dropna() #log returns of closing prices
sigma = returns.std() * np.sqrt(252)
r = 0.05 #assumed risk-free rate


print(f"Spot price: {S0:.2f}, Estimated volatility: {sigma:.4f}")

#a small test set of options
n_test = 10
K_test = np.linspace(S0*0.9, S0*1.1, n_test) #linearly space out strike prices around S0
T_test = np.linspace(0.1, 1.0, n_test) #linearly space out maturities from 0.1 to 1 year
X_test = np.column_stack([np.repeat(S0, n_test), K_test, T_test])

#calculate black-scholes call prices for the test set
BS_prices = [bs_call(S0, K, T, r, sigma) for K, T in zip(K_test, T_test)]

#use trained ML model to predict also
ML_predictions = model.predict(X_test)

#combine into comparison table
results = pd.DataFrame({
    'Strike': K_test,
    'Maturity': T_test,
    'BS_Call': BS_prices,
    'ML_Prediction': ML_predictions
})
results['Diff'] = results['ML_Prediction'] - results['BS_Call']

pd.set_option('display.float_format', '{:.2f}'.format)
print(results)

plt.figure(figsize=(10, 6))
plt.scatter(results['BS_Call'], results['ML_Prediction'], color='blue', label='ML Prediction')
plt.plot([results['BS_Call'].min(), results['BS_Call'].max()],
         [results['BS_Call'].min(), results['BS_Call'].max()],
         color='red', linestyle='--', label='Perfect Match')
plt.xlabel('Black-Scholes Price')
plt.ylabel('ML Predicted Price')
plt.title("Comparison of ML Model vs. Black-Scholes Prices")
plt.legend()
plt.grid(True)

#save to results folder
import os
os.makedirs("results", exist_ok=True)

plt.savefig("results/bs_vs_ml.png", dpi=300, bbox_inches="tight")
plt.show(block=True)
plt.close()