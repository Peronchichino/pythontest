#new approach, not just monte carlo and brownian geometry
# want to look at vix, treasury yield, interest rates, market health, sentiment, beta rates

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns
from scipy import stats
from joblib import Parallel, delayed

portfolio_shares = {
                    #positions here
}

#scenarios: recession, neutral, high_vol, bull_run 
CURRENT_SCENARIO = "neutral"

manual_drift_bias = 0.05 #0.05 = 5% extra annual growth added arbitrarily

T = 1.0 #years
paths = 700000 #throttle up because i have the compute lol
outlier_cap = 100000
risk_free_rate = 0.0426 #current 4.2% yield

tickers = list(portfolio_shares.keys())
macro_tickers = ["^GSPC", "^VIX", "^TNX", "CL=F"] #sp500, vix index, 10yr yield
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

#scenario definition
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

cov_matrix = log_returns.cov() * 252
port_var = np.dot(weights_per_asset.T, np.dot(cov_matrix, weights_per_asset))
port_vol = np.sqrt(port_var)

#marginal contribution to risk (mcr)
mcr = np.dot(cov_matrix, weights_per_asset) / port_vol
#component contribution to risk (crr) as perc
ccr_pct = (weights_per_asset * mcr) / port_vol * 100


# Calculate macro returns to simulate alongside the portfolio
macro_returns = np.log(df[["^GSPC", "CL=F"]] / df[["^GSPC", "CL=F"]].shift(1)).loc[log_returns.index]

#=================================
#monte carlo
#=================================
print(f"Running {paths:,} simulations...")
steps = 252 #trading days
rng = np.random.default_rng(42)

daily_bias = np.log(1 + manual_drift_bias) / 252
log_returns_vals = log_returns.values
macro_returns_vals = macro_returns.values

#use cpu because 12 cores
def run_simulation_batch(batch_size, local_steps, local_returns, local_p_weights, local_bias, local_asset_weights, local_macro_returns):
    #portfolio
    local_rng = np.random.default_rng()
    random_indices = local_rng.choice(len(local_returns), size=(local_steps, batch_size), p=local_p_weights)
    sampled_returns = local_returns[random_indices] + local_bias
    linear_returns = np.exp(sampled_returns) - 1
    daily_port_returns = (linear_returns * local_asset_weights).sum(axis=2)
    port_compounded = np.cumprod(1 + daily_port_returns, axis=0)
    
    #indices
    macro_sampled = local_macro_returns[random_indices]
    macro_sim_cumsum = np.cumsum(macro_sampled, axis=0)
    macro_compounded = np.exp(macro_sim_cumsum)
    
    return port_compounded, macro_compounded

batches = [paths // 12 + (1 if i < paths % 12 else 0) for i in range(12)]

results = Parallel(n_jobs=-1)(delayed(run_simulation_batch)
                              (b, steps, log_returns_vals, prob_weights, daily_bias, weights_per_asset, macro_returns_vals) for b in batches)

port_results, macro_results = zip(*results)
compounded_returns = np.hstack(port_results)
macro_relative_sim = np.hstack(macro_results)

portfolio_val_sim = np.vstack([
    np.full((1, paths), current_val),
    current_val * compounded_returns
])

macro_relative_sim = np.vstack([
    np.ones((1, paths, 2)), # 1.0 multiplier at day 0
    macro_relative_sim
])


# --- SIMULATE BENCHMARKS (S&P 500 & OIL) ---
# Extract full simulation arrays scaled to your starting value
spy_paths = macro_relative_sim[:, :, 0] * current_val
oil_paths = macro_relative_sim[:, :, 1] * current_val

# Calculate the 5th, 50th, and 95th percentiles for SPY
spy_p05 = np.percentile(spy_paths, 5, axis=1)
spy_p50 = np.percentile(spy_paths, 50, axis=1)
spy_p95 = np.percentile(spy_paths, 95, axis=1)

# Calculate the 5th, 50th, and 95th percentiles for Oil
oil_p05 = np.percentile(oil_paths, 5, axis=1)
oil_p50 = np.percentile(oil_paths, 50, axis=1)
oil_p95 = np.percentile(oil_paths, 95, axis=1)

aligned_tnx_ret = np.log(macro_df["^TNX"] / macro_df["^TNX"].shift(1)).dropna().values
aligned_oil_ret = np.log(macro_df["CL=F"] / macro_df["CL=F"].shift(1)).dropna().values

min_len = min(len(aligned_port_ret), len(aligned_tnx_ret), len(aligned_oil_ret))

beta_rates = np.cov(aligned_port_ret[-min_len:], aligned_tnx_ret[-min_len:])[0, 1] / np.var(aligned_tnx_ret[-min_len:])
beta_oil = np.cov(aligned_port_ret[-min_len:], aligned_oil_ret[-min_len:])[0, 1] / np.var(aligned_oil_ret[-min_len:])


#filter outliers above threshold, no need for heavily unrealisitic returns
final_values = portfolio_val_sim[-1]
mask = final_values <= outlier_cap
filtered_finals = final_values[mask]
clean_paths = portfolio_val_sim[:, mask]

sim_total_ret = (filtered_finals / current_val) - 1
exp_return = sim_total_ret.mean()
std_dev = sim_total_ret.std()

#sharpe ratio, sortino ratio (downside risk only)
sharpe = (exp_return - risk_free_rate) / std_dev

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
# VaR Amount: How much money is lost from the start (Positive = Loss, Negative = Gain)
var_95_threshold = np.percentile(filtered_finals, 5)
var_95_amount = current_val - var_95_threshold
cvar_95_threshold = filtered_finals[filtered_finals <= var_95_threshold].mean()
cvar_95_amount = current_val - cvar_95_threshold

#calmar ratio
calmar = exp_return / abs(avg_max_drawdown)

loss_prob = (filtered_finals < current_val).mean() * 100

#===========================
#plots and visuals
#===========================
print("\n" + "="*50)
print(f"PORTFOLIO ANALYTICS REPORT ({CURRENT_SCENARIO.upper()} SCENARIO)")
print("="*50)
print(f"Current Value:    ${current_val:,.2f}")
print(f"Expected Value:   ${filtered_finals.mean():,.2f}")
print(f"Expected Return:  {exp_return*100:.2f}%")
print(f"Interest Rate Beta: {beta_rates:.2f}")
print(f"Crude Oil Beta:     {beta_oil:.2f}")
print("-" * 50)
print(f"RISK METRICS")
print(f"Portfolio Beta:   {portfolio_beta:.2f} (vs S&P 500)")
print(f"Sharpe Ratio:     {sharpe:.2f}")
print(f"Sortino Ratio:    {sortino:.2f}")
print(f"Calmar Ratio:     {calmar:.2f} (Reward vs Pain)")
print(f"Avg Max Drawdown: {avg_max_drawdown*100:.1f}% (Real Pain Expectation)")
print(f"Prob of Loss:     {loss_prob:.1f}%")
print("-" * 50)
print(f"TAIL RISK (95% Confidence)")

# Recalculate these for the print/plot consistency
var_95_threshold = np.percentile(filtered_finals, 5) # The Portfolio Value (e.g., $31k)
var_95_amount = current_val - var_95_threshold       # The Loss Amount (e.g., $1.7k)
cvar_95_threshold = filtered_finals[filtered_finals <= var_95_threshold].mean()
cvar_95_amount = current_val - cvar_95_threshold

sim_skew = stats.skew(sim_total_ret)
sim_kurtosis = stats.kurtosis(sim_total_ret)

# Label logic
var_label = "Loss" if var_95_amount > 0 else "Profit"
print(f"VaR 95%:          ${abs(var_95_amount):,.2f} ({var_label} at Risk)")
print(f"CVaR 95%:         ${cvar_95_amount:,.2f} (Avg outcome if VaR hits)")
print(f"Skewness:         {sim_skew:.2f} (< 0 means fatter downside)")
print(f"Kurtosis:         {sim_kurtosis:.2f} (> 3 means extreme events likely)")
print("-"*50)
print("--- RISK ATTRIBUTION ---")
for i, ticker in enumerate(tickers):
    print(f"{ticker}: Weight {weights_per_asset[i]*100:.1f}% | Risk Contribution: {ccr_pct[i]:.1f}%")
print("="*50)   

# --- PLOTS ---
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 2, width_ratios=[2, 1])
ax1 = fig.add_subplot(gs[:, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 1])
ax4 = fig.add_subplot(gs[2, 1])

# 1. Sample Paths
paths_to_plot = min(500, clean_paths.shape[1])
sample_indices = np.random.choice(clean_paths.shape[1], paths_to_plot, replace=False)
ax1.plot(clean_paths[:, sample_indices], color='darkblue', alpha=0.05, linewidth=0.5)
ax1.axhline(current_val, color='black', linestyle='--', linewidth=1.5, label='Start')
ax1.axhline(outlier_cap, color='red', linestyle='-', linewidth=2, label='Cap Threshold')
ax1.set_title("Sample Paths (Ignored > $100k)\nMethod: Bootstrap + Filter")
ax1.set_ylabel("Portfolio Value ($)")
ax1.yaxis.set_major_locator(MultipleLocator(5000))
# Overlay S&P 500 Simulation Range (Green Cone)
ax1.plot(spy_p50, color='green', linewidth=2, label='Median S&P 500')
ax1.plot(spy_p05, color='green', linestyle=':', linewidth=1)
ax1.plot(spy_p95, color='green', linestyle=':', linewidth=1)
ax1.fill_between(range(len(spy_p50)), spy_p05, spy_p95, color='green', alpha=0.1, label='S&P 500 90% CI')
# Overlay Oil Simulation Range (Orange Cone)
ax1.plot(oil_p50, color='darkorange', linewidth=2, label='Median Oil')
ax1.plot(oil_p05, color='darkorange', linestyle=':', linewidth=1)
ax1.plot(oil_p95, color='darkorange', linestyle=':', linewidth=1)
ax1.fill_between(range(len(oil_p50)), oil_p05, oil_p95, color='darkorange', alpha=0.1, label='Oil 90% CI')
ax1.legend(loc="upper left")

# 2. Distribution (Histogram)
sns.histplot(filtered_finals, bins=100, kde=True, color='purple', ax=ax2, stat='density')
ax2.axvline(current_val, color='k', linestyle='--')
ax2.axvline(var_95_threshold, color='red', linestyle='-', label=f'VaR 95%: ${var_95_threshold:,.0f}')
ax2.set_title("Distribution of Returns at Year End")
ax2.legend()

# 3. Correlation Matrix
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=False, ax=ax3)
ax3.set_title("Asset Correlation Matrix (Last 5Y)")

# 4. Drawdown Chart (FIXED)
avg_drawdown_curve = drawdowns.mean(axis=1) * 100

ax4.plot(avg_drawdown_curve, color='red', label='Avg Drawdown')
ax4.fill_between(range(len(avg_drawdown_curve)), avg_drawdown_curve, 0, color='red', alpha=0.3)
ax4.set_title("Expected Drawdown Trajectory")
ax4.set_ylabel("Drawdown %")
ax4.legend()

#window manager
manager = plt.get_current_fig_manager()
manager.full_screen_toggle()
plt.tight_layout()
plt.show()
