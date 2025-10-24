import yfinance as yf
import matplotlib.pyplot as plt

ticker = "MSFT"

data = yf.download(ticker, period="5d")

print(data.head())

plt.figure(figsize=(10,5))
plt.plot(data.index, data["Close"], label=["Close Price"])
plt.title(f"{ticker} Stoc price (Last month)")
plt.xlabel("Time")
plt.ylabel("Price(USD)")
plt.legend()
plt.grid(True)
plt.show()