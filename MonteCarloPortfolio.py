import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

portfolio_shares = {"AMD":20,
                    "BDTX":200, 
                    "BLK":10, #was 14.76
                    "CRML":150, #was 100
                    "GOOGL":7, 
                    "NAK":200, 
                    "NB":50, 
                    "NVDA":21.47,
                    "PPTA":50, #was 15
                    "TMC":200, #was 50
                    "UUUU":50, #was 25
                    "ASML":5 #dont have 
}

T = 1.0 #time horizon in years
paths = 100000
risk_free_rate = 0.04 #4% treasury bill yield, need to update to get more accuracy
seed = 42

#fetch data
tickers = list(portfolio_shares.keys())
data = yf.download(tickers, period="3y", progress=False, auto_adjust=False)

if isinstance(data.columns, pd.MultiIndex):
    try: 
        df = data["Adj Close"]
    except KeyError:
        df = data["Close"]
else:
    col = "Adj Close" if "Adj Close" in data.columns else "Close"
    df = data[col]
    
df = df.dropna(axis=1, how='all')
valid_tickers = list(df.columns)

df = df.dropna(how="any")

shares_list = []
for t in valid_tickers:
    shares_list.append(portfolio_shares[t])
    
shares_array = np.array(shares_list)

#calculate stats
#log returns are additive, making them easier for GBM math
#og_returns = np.log(df / df.shift(1)).dropna()

#drift and volatility (sigma) calculation
#mu_daily = log_returns.mean().values
#cov_daily = log_returns.cov().values
S0 = df.iloc[-1].values #starting prices
current_portfolio_value = np.sum(S0 * shares_array)

log_returns = np.log(df / df.shift(1)).dropna()
mu = log_returns.mean().values
sigma_cov = log_returns.cov().values

#simulation
steps = 252
dt = T / steps
rng = np.random.default_rng(seed) #fixed seed


#cholesky decomposition for correlation
#daily covariance matrix to correlate the random shocks
L = np.linalg.cholesky(sigma_cov)

#generate random shocks
Z = rng.standard_normal((steps, paths, len(valid_tickers)))

#correlate the shocks and drift
#reshape to (N, assets) -> multiply -> reshape back
Z_corr = (Z.reshape(steps * paths, len(valid_tickers)) @ L.T).reshape(steps, paths, len(valid_tickers))
drift = mu[None, None, :]

#generate prices
inc = drift +Z_corr
log_ret_sim = np.cumsum(inc, axis=0)
log_ret_sim = np.vstack([np.zeros((1, paths, len(valid_tickers))), log_ret_sim])
price_sim = S0[None, None, :] * np.exp(log_ret_sim)

#portfolio valuation
portfolio_val = (price_sim * shares_array[None, None, :]).sum(axis=2)

#risk metrics
final_vals = portfolio_val[-1]
sim_returns = (final_vals / current_portfolio_value) -1

var_95_val = np.percentile(final_vals, 5)
var_95_loss = current_portfolio_value - var_95_val
cvar_val = final_vals[final_vals <= var_95_val].mean()
prob_loss = (final_vals < current_portfolio_value).mean() * 100

#sharpe ratio
expected_return = sim_returns.mean()
volatility = sim_returns.std()
sharpe_ratio = (expected_return - risk_free_rate) / volatility


print("=" * 45)
print(f"PORTFOLIO SIMULATION REPORT")
print("=" * 45)
print(f"Start Value:      ${current_portfolio_value:,.2f}")
print(f"Exp. End Value:   ${np.mean(final_vals):,.2f}")
print(f"Exp. Return:      {expected_return*100:.1f}%")
print("-" * 45)
print(f"RISK ANALYSIS:")
print(f"Sharpe Ratio:     {sharpe_ratio:.2f} (Reward/Risk)")
print(f"Prob of Loss:     {prob_loss:.1f}%")
print(f"VaR (95%):        -${var_95_loss:,.2f} (Worst 5% Outcome)")
print(f"CVaR (95%):       ${cvar_val:,.2f} (Avg Crash Outcome)")
print("=" * 45)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Path Plot
ax1.plot(portfolio_val[:, :200], color='tab:blue', alpha=0.15, linewidth=0.8)
ax1.axhline(current_portfolio_value, color='k', linestyle='--', linewidth=1.5, label='Start')
ax1.set_title(f"200 Random Paths (1 Year)\nCurrent: ${current_portfolio_value:,.0f}")
ax1.set_ylabel("Portfolio Value ($)")
ax1.set_xlabel("Trading Days")
ax1.grid(True, alpha=0.2)

# Distribution Plot
ax2.hist(final_vals, bins=50, color='tab:blue', alpha=0.7, density=True)
ax2.axvline(current_portfolio_value, color='k', linestyle='--', linewidth=2, label='Start')
ax2.axvline(var_95_val, color='tab:red', linestyle='-', linewidth=2, label=f'VaR 95%: ${var_95_val:,.0f}')
ax2.axvline(cvar_val, color='darkred', linestyle=':', linewidth=2, label=f'CVaR: ${cvar_val:,.0f}')

ax2.set_title(f"Outcome Distribution\nSharpe: {sharpe_ratio:.2f} | Loss Prob: {prob_loss:.1f}%")
ax2.set_xlabel("Portfolio Value ($)")
ax2.legend()
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()

