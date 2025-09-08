from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np

#generate data
n_samples = 5000
S0 = np.random.uniform(50, 150, n_samples)
K = np.random.uniform(50, 150, n_samples)
T = np.random.uniform(0.1, 2, n_samples)
r = 0.05
sigma = 0.2
#this creates 5000 random combinations of option parameters S, K, T
#risk-free rate and volatility are kept constant

#Monte Carlo to calculate option prices
def monte_carlo_call(S0, K, T, r, sigma, n_sim=10000):
    Z = np.random.standard_normal(n_sim)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoff = np.maximum(ST - K, 0)
    return np.exp(-r*T) * np.mean(payoff)
#this runs 10,000 simulations for each option to estimate its price
#computes the european call option payoff at maturity 

#stores these prices in y which is the target label for the ML model
y = np.array([monte_carlo_call(S0[i], K[i], T[i], r, sigma) for i in range(n_samples)])
X = np.column_stack([S0, K, T]) #input features

#to split data into training and testing sets. 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#trains a random forest regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
#this teaches the model to learn the r/s between S, K, T, and the option price

#predicts option price
pred = model.predict(X_test)
print(pred[:5]) #prints the first 5 predicted prices

#monte carlo + ML for option pricing
#this model trains a machine learning model to learn the pricing function