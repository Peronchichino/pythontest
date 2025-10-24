import yfinance as yf

msft = yf.Ticker("MSFT")

msft_balance = msft.get_balance_sheet()

#print(msft)
print(msft_balance)