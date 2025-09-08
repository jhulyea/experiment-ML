import yfinance as yf
import os

#this is just to make sure data folder exists 
os.makedirs("data", exist_ok=True)  

#downloading apple stock in the past two yrs
data = yf.download("AAPL", start="2023-01-01", end="2025-08-01")

#save as csv
data.to_csv("data/historical_stock_data.csv")

print("Data downloaded and saved to data/historical_stock_data.csv")