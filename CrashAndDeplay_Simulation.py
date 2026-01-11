import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION ---
# Your "Bulletproof" Portfolio
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

# --- STRESS TEST SETTINGS ---
crash_duration = 60      # The crash lasts ~3 months (60 trading days)
crash_vol_multiplier = 2.5 # Volatility explodes (2.5x normal)
crash_correlation = 0.9  # EVERYTHING moves together (Panic correlation)
cash_on_side = 10000     # <--- ENTER YOUR EXTRA CASH AMOUNT HERE
deploy_day = 60          # We buy the dip exactly at the end of the crash

# Normal Growth Caps (for the recovery phase)
recovery_caps = { "NVDA": 0.20, "AMD": 0.20, "GOOGL": 0.15 }

# Simulation Settings
paths = 50000   # Fast & precise
seed = 42

# --- 2. DATA FETCHING ---
tickers = list(portfolio_shares.keys())
print(f"Loading data for Stress Test...")
data = yf.download(tickers, period="3y", progress=False, auto_adjust=False)

if isinstance(data.columns, pd.MultiIndex):
    try: df = data["Adj Close"]
    except KeyError: df = data["Close"]
else:
    col = "Adj Close" if "Adj Close" in data.columns else "Close"
    df = data[col]

df = df.dropna(axis=1, how='all').dropna(how="any")
valid_tickers = list(df.columns)
shares_array = np.array([portfolio_shares[t] for t in valid_tickers])
S0 = df.iloc[-1].values
start_val = np.sum(S0 * shares_array)

# --- 3. CONSTRUCTING THE TWO REGIMES ---
# A. Normal Regime (Historical Stats)
log_ret = np.log(df / df.shift(1)).dropna()
mu_normal = log_ret.mean().values
cov_normal = log_ret.cov().values
std_normal = np.sqrt(np.diag(cov_normal))

# Apply Growth Caps to Normal Drift
for i, t in enumerate(valid_tickers):
    if t in recovery_caps:
        mu_normal[i] = np.log(1 + recovery_caps[t]) / 252

# B. Crash Regime (Synthetic Panic)
# 1. Create Panic Covariance: Set all correlations to 'crash_correlation'
# Cov_ij = rho * sigma_i * sigma_j
cov_crash = np.outer(std_normal * crash_vol_multiplier, std_normal * crash_vol_multiplier) * crash_correlation
np.fill_diagonal(cov_crash, (std_normal * crash_vol_multiplier)**2) # Diagonal is just variance

# 2. Create Panic Drift: Force heavy negative trend
# Tech/Miners drop hard (-50% annualized), Anchors drop less (-20% annualized)
mu_crash = np.zeros_like(mu_normal)
for i, t in enumerate(valid_tickers):
    if t in ["JNJ", "BRK-B", "BLK", "WMT", "KO"]:
        mu_crash[i] = np.log(1 - 0.20) / 252 # Anchors drop 20% annualized
    else:
        mu_crash[i] = np.log(1 - 0.60) / 252 # Risky assets drop 60% annualized

# --- 4. RUNNING THE SIMULATION (Step-by-Step) ---
print(f"Simulating 'Liquidity Crisis' Scenario ({paths} runs)...")
rng = np.random.default_rng(seed)

# Pre-calculate Cholesky Decompositions
L_crash = np.linalg.cholesky(cov_crash)
L_normal = np.linalg.cholesky(cov_normal)

# Arrays to hold price paths
# Shape: (Days, Paths, Assets)
sim_prices = np.zeros((252, paths, len(valid_tickers)))
sim_prices[0] = S0 # Set start price

current_prices = np.tile(S0, (paths, 1))

for t in range(1, 252):
    # Generating Shocks
    Z = rng.standard_normal((paths, len(valid_tickers)))
    
    if t <= crash_duration:
        # CRASH REGIME
        # shock = Z @ L_crash.T
        shock = Z @ L_crash.T
        drift = mu_crash
    else:
        # RECOVERY REGIME
        shock = Z @ L_normal.T
        drift = mu_normal
        
    # Update Prices (Geometric Brownian Motion Step)
    # P_t = P_t-1 * exp(drift + shock)
    current_prices = current_prices * np.exp(drift + shock)
    sim_prices[t] = current_prices

# --- 5. PORTFOLIO VALUE & CASH DEPLOYMENT ---
# Calculate Value WITHOUT Cash Injection first (to see the pain)
val_no_inject = (sim_prices * shares_array[None, None, :]).sum(axis=2)

# Calculate Value WITH Cash Injection
# At day 'deploy_day', we buy more shares of everything (proportional to current weights)
# Or simpler: we just add cash value to the portfolio total? 
# To be accurate, we convert Cash -> Shares at Day 60 prices.

# 1. Get prices at Day 60
prices_at_dip = sim_prices[deploy_day] # (Paths, Assets)

# 2. Determine weights to buy (maintain current portfolio ratios)
# We buy a "slice" of the portfolio basket
# Basket Cost = sum(Price * Shares) / Total_Shares? No.
# We distribute cash proportionally to the current value of positions?
# Let's assume we buy the *same mix* of shares we currently hold.
portfolio_mix_weights = (S0 * shares_array) / start_val
cash_per_asset = cash_on_side * portfolio_mix_weights
new_shares_per_path = cash_per_asset / prices_at_dip # (Paths, Assets)

# 3. Create a "Share Count Path"
# Days 0-60: Original Shares
# Days 61+: Original + New Shares
shares_path = np.zeros((252, paths, len(valid_tickers)))
shares_path[:deploy_day] = shares_array
shares_path[deploy_day:] = shares_array + new_shares_per_path

# 4. Final Portfolio Value Path
val_injected = (sim_prices * shares_path).sum(axis=2)

# --- 6. REPORTING ---
final_vals = val_injected[-1]
min_vals = val_no_inject.min(axis=0) # Worst point before injection
avg_drawdown = (min_vals.mean() - start_val) / start_val

print("=" * 50)
print(f"STRESS TEST: 'THE LIQUIDITY CRISIS'")
print(f"Scenario: 3-Month Panic (-60% annualized trend) -> Recovery")
print("=" * 50)
print(f"Initial Portfolio:  ${start_val:,.0f}")
print(f"Cash Deployed:      ${cash_on_side:,.0f} (at Day {deploy_day})")
print("-" * 50)
print(f"THE PAIN (Before Cash Injection):")
print(f"Avg Crash Bottom:   ${min_vals.mean():,.0f}")
print(f"Avg Drawdown:       {avg_drawdown*100:.1f}%")
print("-" * 50)
print(f"THE RECOVERY (End of Year):")
print(f"Avg Final Value:    ${final_vals.mean():,.0f}")
print(f"Prob of Profit:     {(final_vals > (start_val + cash_on_side)).mean()*100:.1f}%")
print(f"Return on Cash:     {((final_vals.mean() - min_vals.mean()) / cash_on_side - 1)*100:.1f}% (Est)")
print("=" * 50)

# --- 7. VISUALS ---
plt.figure(figsize=(12, 6))

# Plot "No Injection" mean path
plt.plot(val_no_inject.mean(axis=1), color='red', linestyle='--', label='Hold Only (No Cash)')

# Plot "With Injection" mean path
plt.plot(val_injected.mean(axis=1), color='green', linewidth=2, label='With Cash Injection')

# Plot 50 random paths (Injected)
plt.plot(val_injected[:, :50], color='green', alpha=0.1)

plt.axvline(deploy_day, color='k', linestyle=':', label='Cash Deployed')
plt.title(f"Crash & Deploy Simulation: +${cash_on_side:,.0f} Injection")
plt.ylabel("Portfolio Value ($)")
plt.xlabel("Trading Days")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()