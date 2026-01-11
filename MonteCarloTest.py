#monte carlo simulation
#using geometric brownian motion
import numpy as np, pandas as pd, yfinance as yf, matplotlib.pyplot as plt

tickers = ["AMD", "BDTX", "BLK", "NVDA", "TMC", "CRML", "GOOGL", "NAK", "NB", "PPTA", "UUUU"]

#robust price fetch
data = yf.download(tickers, period="3y", progress=False, auto_adjust=False)

if isinstance(data.columns, pd.MultiIndex):
    df = data["Adj Close"].dropna(how="any")
else:
    if set(tickers).issubset(data.columns):
        df = data[tickers].dropna(how="any")
    elif "Close" in data.columns:
        df = data["Close"].dropna(how="any")
    else:
        raise RuntimeError("Unexpected yf.download format; inspect data.columns")

logr = np.log(df).diff().dropna() #daily log returns
S0 = df.iloc[-1].values.astype(float) #latest prices

T, steps, paths, seed = 1.0, 252, 5000, 123 #1 year, 252 trading days, 5000 runs of sim, randnum gen seed
dt = T / steps

mu = (logr.mean().values / dt) + 0.5 * (logr.std(ddof=1).values * np.sqrt(252))**2
Sigma = logr.cov().values * 252 #anual coveriance

rng = np.random.default_rng(seed)
L = np.linalg.cholesky(Sigma * dt)
Z = rng.standard_normal((steps, paths, len(S0)))
Zf = Z.reshape(steps*paths, len(S0)) @ L.T
corr = Zf.reshape(steps, paths, len(S0))

drift = (mu - 0.5 * np.diag(Sigma)) * dt
inc = drift[None, None,:] + corr
logp = np.vstack([np.zeros((1, paths, len(S0))), np.cumsum(inc, axis=0)])

assets = S0[None, None,:] * np.exp(logp)

w = np.repeat(1/len(S0), len(S0))
portfolio = (assets * w.reshape(1, 1, -1)).sum(axis=2)

ST = portfolio[-1]
p5, p50, p95 = np.percentile(ST, [5, 50, 95])
print(f"Start: {np.sum(S0*w):.2f} | 5%={p5:.2f} median={p50:.2f} 95%={p95:.2f}")

#visual
plt.figure(figsize=(9,4))
for i in rng.choice(paths, size=min(200, paths), replace=False):
    plt.plot(portfolio[:,i], color="tab:blue", alpha=0.4, linewidth=0.6)
plt.axhline(np.sum(S0*w), color="k", linestyle="--")
plt.title("My Portfolio"); plt.xlabel("Day"); plt.ylabel("Value")
plt.tight_layout(); plt.show()