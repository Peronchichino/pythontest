import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

portfolio_shares = {"AMD":20,
                    "BDTX":200, 
                    "BLK":10, #was 14.76
                    "CRML":100, #was 100
                    "GOOGL":7, 
                    "NAK":200, 
                    "NB":50, 
                    "NVDA":21.47,
                    "PPTA":25, #was 15
                    "TMC":100, #was 50
                    "UUUU":25, #was 25
                    #some other anchor, i have 5k dollars sitting empty in my portfolio
                    "JNJ":10,
                    "BRK-B": 10
}

#scenario override for tech because of personal opinion
manual_growth_caps = {
    "NVDA": 0.30,
    "AMD": 0.20,
    "GOOGL": 0.15
}

T = 1.0 #time horizon in years
paths = 250000
risk_free_rate = 0.04 #4% treasury bill yield, need to update to get more accuracy
seed = 42
outlier_cap = 100000 #ignore all runs above 100k, not realistic

tickers = list(portfolio_shares.keys())
print(f"Fetching data for {len(tickers)} assets...")
data = yf.download(tickers, period="3y", progress=False, auto_adjust=False)

if isinstance(data.columns, pd.MultiIndex):
    try:
        df = data["Adj Close"]
    except KeyError:
        df = data["Close"]
else:
    col = "Adj Close" if "Adj Close" in data.columns else "Close"
    df = data[col]

df = df.dropna(axis=1, how='all').dropna(how="any")
valid_tickers = list(df.columns)

shares_array = np.array([portfolio_shares[t] for t in valid_tickers])
S0 = df.iloc[-1].values
current_val = np.sum(S0 * shares_array)

#bootstrap prep
hist_log_ret = np.log(df / df.shift(1)).dropna()
n_hist_days = len(hist_log_ret)

#center data and apply caps
mean_hist = hist_log_ret.mean().values
centered_ret = hist_log_ret.values - mean_hist
target_drift = mean_hist.copy()

print("-" * 40)
print("Applying Growth Rate Caps:")
for i, ticker in enumerate(valid_tickers):
    if ticker in manual_growth_caps:
        ann_target = manual_growth_caps[ticker]
        daily_target = np.log(1 + ann_target) / 252
        target_drift[i] = daily_target
        print(f"  > {ticker}: Forced to {ann_target*100:.1f}% Annual")

#bootstrap engine
print(f"Running {paths:,} simulations (Bootstrap)...")
steps = 252
rng = np.random.default_rng(seed)

random_day_indices = rng.integers(0, n_hist_days, size=(steps, paths))
sampled_returns = centered_ret[random_day_indices]
sim_log_inputs = sampled_returns + target_drift[None, None, :]
log_ret_sim = np.cumsum(sim_log_inputs, axis=0)
log_ret_sim = np.vstack([np.zeros((1, paths, len(valid_tickers))), log_ret_sim])
price_sim = S0[None, None, :] * np.exp(log_ret_sim)
portfolio_val = (price_sim * shares_array[None, None, :]).sum(axis=2)

#filter runs aka any above 100k, too implausible
final_values_raw = portfolio_val[-1]
valid_mask = final_values_raw <= outlier_cap

#apply mask
filtered_portfolio_val = portfolio_val[:, valid_mask]
filtered_final_vals = filtered_portfolio_val[-1]

dropped_count = paths - len(filtered_final_vals)
pct_dropped = (dropped_count / paths) * 100

print("-" * 40)
print(f"OUTLIER FILTER APPLIED:")
print(f"Removed {dropped_count:,} runs ({pct_dropped:.1f}%) that ended above ${outlier_cap:,.0f}")
print(f"Remaining Runs: {len(filtered_final_vals):,}")
print("-" * 40)

#metrics
sim_returns = (filtered_final_vals / current_val) - 1

var_95_val = np.percentile(filtered_final_vals, 5)
var_95_loss = current_val - var_95_val
cvar_val = filtered_final_vals[filtered_final_vals <= var_95_val].mean()
prob_loss = (filtered_final_vals < current_val).mean() * 100
sharpe = (sim_returns.mean() - risk_free_rate) / sim_returns.std()

#report
print("=" * 45)
print(f"CONSERVATIVE SCENARIO (Capped @ ${outlier_cap/1000:.0f}k)")
print("=" * 45)
print(f"Start Value:      ${current_val:,.2f}")
print(f"Exp. End Value:   ${np.mean(filtered_final_vals):,.2f}")
print(f"Exp. Return:      {sim_returns.mean()*100:.1f}%")
print("-" * 45)
print(f"RISK METRICS (Excluding 'Lucky' Outcomes):")
print(f"Prob of Loss:     {prob_loss:.1f}%")
print(f"Sharpe Ratio:     {sharpe:.2f}")
print(f"VaR (95%):        ${var_95_loss:,.2f} (Loss Amount)")
print(f"CVaR (95%):       ${cvar_val:,.2f} (Avg Crash Value)")
print("=" * 45)

#visuals
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

#paths
display_paths = min(500, filtered_portfolio_val.shape[1])
ax1.plot(filtered_portfolio_val[:, :display_paths], color='tab:blue', alpha=0.1, linewidth=0.5)
ax1.axhline(current_val, color='k', linestyle='--', label='Start')
ax1.axhline(outlier_cap, color='r', linestyle='-', linewidth=2, label='Cap Threshold')
ax1.set_title(f"Sample Paths (Ignored > ${outlier_cap/1000:.0f}k)\nMethod: Bootstrap + Filter")
ax1.set_ylabel("Portfolio Value ($)")
ax1.legend()

#distribution
ax2.hist(filtered_final_vals, bins=100, color='tab:purple', alpha=0.7, density=True)
ax2.axvline(current_val, color='k', linestyle='--', linewidth=2, label='Start')
ax2.axvline(var_95_val, color='tab:red', linestyle='-', linewidth=2, label=f'VaR 95%: ${var_95_val:,.0f}')
ax2.set_title(f"Outcome Distribution (Truncated)\nLoss Prob: {prob_loss:.1f}%")
ax2.set_xlabel("Final Value ($)")
ax2.legend()

plt.tight_layout()
plt.show()