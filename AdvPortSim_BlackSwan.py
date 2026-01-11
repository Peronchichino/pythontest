#adding black swan event (10+% market crash) to AdvPortSim_test.py monte carlo simulation
#new approach, not just monte carlo and brownian geometry
# want to look at vix, treasury yield, interest rates, market health, sentiment

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

portfolio_shares = {
    "AMD":20,
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

CURRENT_SCENARIO = "neutral"

manual_drift_bias = 0.00 #0.05 = 5% extra annual growth added arbitrarily

T = 1.0 #years
paths = 100000
outlier_cap = 150000
risk_free_rate = 0.042 #current 4.2% yield

tickers = list(portfolio_shares.keys())
macro_tickers = ["^GSPC", "^VIX", "^TNX"] #sp500, vix index, 10yr yield
all_tickers = tickers + macro_tickers

print(f"Fetching data for assets...")
data = yf.download(all_tickers, period="5y", progress=False, auto_adjust=False)

if isinstance(data.columns, pd.MultiIndex):
    try:
        df = data["Adj Close"]
    except KeyError:
        df = data["Close"]
else:
    col = "Adj Close" if "Adj Close" in data.columns else "Close"
    df = data[col]
    
#fill missing macro data
df[macro_tickers] = df[macro_tickers].ffill()
df = df.dropna(how="any")

#seperate stock and macro data
stock_df = df[tickers].copy()
macro_df = df[macro_tickers].copy()

#feature engineering and regimes
#log returns
log_returns = np.log(stock_df / stock_df.shift(1)).dropna()
macro_df = macro_df.loc[log_returns.index] #align dates

#vix
vix_median = macro_df["^VIX"].median()
is_high_vol = macro_df["^VIX"] > vix_median

#rate regime (yield/rate)
rate_change = macro_df["^TNX"].diff(20) #20 trading days
is_rising_rates = rate_change > 0

#market trend, moving average
spy_ma_50 = macro_df["^GSPC"].rolling(50).mean()
is_bull_trend = macro_df["^GSPC"] > spy_ma_50

#define probability weights for sampling based on scenario
#default is 1
weights = np.ones(len(log_returns))

print("-"*40)
print(f"Apply Macro Regime: {CURRENT_SCENARIO.upper()}")

if CURRENT_SCENARIO == "high_vol":
    weights[is_high_vol] *= 3.0
    print(f" > Overweighting High Vol days (VIX > {vix_median:.1f})")
elif CURRENT_SCENARIO == "rising_rates":
    weights[is_rising_rates] *= 3.0
    print(f" > Overweighting days where Rates rose")
elif CURRENT_SCENARIO == "recession":
    mask = is_high_vol & (~is_bull_trend)
    weights[mask] *= 5.0
    print(f" > Overweighting High Vol + downward trend days")
elif CURRENT_SCENARIO == "bull_run":
    mask = (~is_high_vol) & (is_bull_trend)
    weights[mask] *= 3.0
    print(f" > Overweighting low vol + bull trend days")

#normalize weights to sum to 1
prob_weights = weights / weights.sum()

#portfolio prep and analytics
shares_array = np.array([portfolio_shares[t] for t in tickers])
S0 = stock_df.iloc[-1].values
current_val = np.sum(S0 * shares_array)

#portfolio history
weights_per_asset = (S0 * shares_array) /current_val
port_hist_ret = (log_returns * weights_per_asset).sum(axis=1)

#calc beta against sp500
spy_ret_raw = np.log(macro_df["^GSPC"] / macro_df["^GSPC"].shift(1))

alignment_df = pd.DataFrame({
    'Port':port_hist_ret,
    'SPY':spy_ret_raw
}).dropna()

aligned_port_ret = alignment_df['Port'].values
aligned_spy_ret = alignment_df['SPY'].values


#beta
covariance = np.cov(aligned_port_ret, aligned_spy_ret)[0, 1]
variance = np.var(aligned_spy_ret)
portfolio_beta = covariance / variance

#correlation matrix
corr_matrix = log_returns.corr()

#monte carlo
print(f"Running {paths:,} simulations...")
steps = 252 #trading days
rng = np.random.default_rng(42)

random_indices = rng.choice(len(log_returns), size=(steps, paths), p=prob_weights)
sampled_returns = log_returns.values[random_indices]

#apply manual drift bias (pres cycle, news sentiment)
daily_bias = np.log(1 + manual_drift_bias) /252
sampled_returns += daily_bias

#black swan event injection
#0.0005 daily prob approx =12% change of crash occurring per year per stock
CRASH_PROB_DAILY = 0.0002
CRASH_SEVERITY = -0.5 #-50% drop, denied permits, tarifs, etc

#generate crash market
crash_mask = rng.choice([0, 1], size=sampled_returns.shape, p=[1 - CRASH_PROB_DAILY, CRASH_PROB_DAILY])
crash_log_ret = np.log(1 + CRASH_SEVERITY)

#apply shocks
sampled_returns = np.where(crash_mask == 1, crash_log_ret, sampled_returns)
total_crashes = np.sum(crash_mask)
print(f" > BLACK SWAN INJECTED: {total_crashes} individual stock crashes simulated")
print(f"    (Modeling a {CRASH_SEVERITY*100:.0f}% drop event for single asset)")

#aggregate returns
sim_log_cumsum = np.cumsum(sampled_returns, axis=0)
sim_log_cumsum = np.vstack([np.zeros((1, paths, len(tickers))), sim_log_cumsum])

#calc prices and value
price_sim = S0[None, None, :] * np.exp(sim_log_cumsum)
portfolio_val_sim = (price_sim * shares_array[None, None, :]).sum(axis=2)

#filter outliers
final_values = portfolio_val_sim[-1]
mask = final_values <= outlier_cap
filtered_finals = final_values[mask]
clean_paths = portfolio_val_sim[:, mask]

#advanced metrics
sim_total_ret = (filtered_finals / current_val) - 1
exp_return = sim_total_ret.mean()
std_dev = sim_total_ret.std()

#sharpe ratio
sharpe = (exp_return - risk_free_rate) / std_dev

#sortino ratio, downside risk only
downside_returns = sim_total_ret[sim_total_ret < 0]
downside_std = downside_returns.std() if len(downside_returns) > 0 else 1e-6
sortino = (exp_return - risk_free_rate) / downside_std

#max drawdown
#calc MDD for expected path
running_max = np.maximum.accumulate(clean_paths, axis=0)
drawdowns = (clean_paths - running_max) / running_max
max_drawdowns_per_path = drawdowns.min(axis=0) # Best is 0, Worst is -0.5 etc.
avg_max_drawdown = max_drawdowns_per_path.mean()

#var and cvar
var_95_threshold = np.percentile(filtered_finals, 5)
# VaR Amount: How much money is lost from the start (Positive = Loss, Negative = Gain)
var_95_amount = current_val - var_95_threshold
cvar_95_threshold = filtered_finals[filtered_finals <= var_95_threshold].mean()
cvar_95_amount = current_val - cvar_95_threshold

loss_prob = (filtered_finals < current_val).mean() * 100


#plots and visuals
print("\n" + "="*50)
print(f"PORTFOLIO ANALYTICS REPORT ({CURRENT_SCENARIO.upper()} SCENARIO)")
print("="*50)
print(f"Current Value:    ${current_val:,.2f}")
print(f"Expected Value:   ${filtered_finals.mean():,.2f}")
print(f"Expected Return:  {exp_return*100:.2f}%")
print("-" * 50)
print(f"RISK METRICS")
print(f"Portfolio Beta:   {portfolio_beta:.2f} (vs S&P 500)")
print(f"Sharpe Ratio:     {sharpe:.2f}")
print(f"Sortino Ratio:    {sortino:.2f}")
print(f"Avg Max Drawdown: {avg_max_drawdown*100:.1f}% (Real Pain Expectation)")
print(f"Prob of Loss:     {loss_prob:.1f}%")
print("-" * 50)
print(f"TAIL RISK (95% Confidence)")

# Recalculate these for the print/plot consistency
var_95_threshold = np.percentile(filtered_finals, 5) # The Portfolio Value (e.g., $31k)
var_95_amount = current_val - var_95_threshold       # The Loss Amount (e.g., $1.7k)
cvar_95_threshold = filtered_finals[filtered_finals <= var_95_threshold].mean()
cvar_95_amount = current_val - cvar_95_threshold

# Label logic
var_label = "Loss" if var_95_amount > 0 else "Profit"
print(f"VaR 95%:          ${abs(var_95_amount):,.2f} ({var_label} at Risk)")
print(f"CVaR 95%:         ${cvar_95_amount:,.2f} (Avg outcome if VaR hits)")
print("="*50)

# --- PLOTS ---
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 2)

# 1. Simulation Cone
ax1 = fig.add_subplot(gs[0, 0])
p05 = np.percentile(clean_paths, 5, axis=1)
p50 = np.percentile(clean_paths, 50, axis=1)
p95 = np.percentile(clean_paths, 95, axis=1)

ax1.plot(p50, color='blue', label='Median Path')
ax1.fill_between(range(len(p50)), p05, p95, color='blue', alpha=0.1, label='95% Confidence Interval')
ax1.axhline(current_val, color='k', linestyle='--', label='Start')
ax1.set_title(f"Projected Portfolio Range ({CURRENT_SCENARIO})")
ax1.set_ylabel("Portfolio Value ($)")
ax1.legend()

# 2. Distribution (Histogram)
ax2 = fig.add_subplot(gs[0, 1])
sns.histplot(filtered_finals, bins=100, kde=True, color='purple', ax=ax2, stat='density')
ax2.axvline(current_val, color='k', linestyle='--')
# FIX: Plot the Threshold Value ($31k), not the Loss Amount ($1.7k)
ax2.axvline(var_95_threshold, color='red', linestyle='-', label=f'VaR 95%: ${var_95_threshold:,.0f}')
ax2.set_title("Distribution of Returns at Year End")
ax2.legend()

# 3. Correlation Matrix
ax3 = fig.add_subplot(gs[1, 0])
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=False, ax=ax3)
ax3.set_title("Asset Correlation Matrix (Last 5Y)")

# 4. Drawdown Chart (FIXED)
ax4 = fig.add_subplot(gs[1, 1])
# FIX: Calculate the MEAN curve of all drawdowns (Collapse 2D -> 1D)
avg_drawdown_curve = drawdowns.mean(axis=1) * 100

ax4.plot(avg_drawdown_curve, color='red', label='Avg Drawdown')
ax4.fill_between(range(len(avg_drawdown_curve)), avg_drawdown_curve, 0, color='red', alpha=0.3)
ax4.set_title("Expected Drawdown Trajectory")
ax4.set_ylabel("Drawdown %")
ax4.legend()

plt.tight_layout()
plt.show()