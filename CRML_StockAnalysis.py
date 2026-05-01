import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns
from scipy import stats

ticker = 'CRML'
days_to_forecast = 252
num_simulations = 10000
target_price = 20.00

data = yf.download(ticker, period='1y')

close_price = data['Close'].squeeze()
returns = close_price.pct_change().dropna()

last_price = float(close_price.iloc[-1])

#mu = returns.mean()
mu = 0.0
sigma = returns.std()

simulations_df = pd.DataFrame()
all_sims_data = []

#CRML support zones
sup_high = 10.50
sup_low = 9.50
buy_pressure = 0.02 #2% daily upward bias when in supp zone
panic_sell = 0.015 #1.5% daily downward bias if supp breaks

for i in range(num_simulations):
    prices = [last_price]
    for d in range(days_to_forecast):
        shock = np.random.normal()
        drift = mu - 0.5 * sigma**2
        curr_price = prices[-1]
        
        #supp zone
        if sup_low <= curr_price <= sup_high:
            drift += buy_pressure
        elif curr_price < sup_low:
            drift += panic_sell
        
        change = np.exp(drift + sigma * shock)
        prices.append(prices[-1] * change)
        
    all_sims_data.append(prices)
    
simulations_df = pd.DataFrame(all_sims_data).T

# 6. Success Probability
ending_prices = simulations_df.iloc[-1]
prob_of_target = (ending_prices >= target_price).sum() / num_simulations * 100
avg_ending = ending_prices.mean()

print(f"--- Analysis for {ticker} ---")
print(f"Current Price: ${last_price:.2f}")
print(f"Probability of hitting ${target_price}: {prob_of_target:.2f}%")
print(f"Average projected price in 1 year: ${avg_ending:.2f}")
print(f"95% Confidence Interval: ${np.percentile(ending_prices, 2.5):.2f} to ${np.percentile(ending_prices, 97.5):.2f}")

plt.figure(figsize=(12,6))
plt.plot(simulations_df, lw=0.5, alpha=0.3, color='gray')
plt.axhline(y=target_price, color='r', linestyle='--', label=f'Target: ${target_price}')
plt.axhline(y=last_price, color='blue', label=f'Current: ${last_price:.2f}')
plt.title(f'Monte Carlo Simulation: {ticker} (1 Year Outlook)')
plt.xlabel('Days')
plt.ylabel('Price ($)')
plt.legend()
plt.show()


