# download the data from
# www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs
import os

tickers = [
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "GOOG",
    "FB",
    "INTC",
    "CSCO",
    "CMCSA",
    "PEP",
    "ADBE",
    "NFLX",
    "NVDA",
    "COST",
    "AMGN",
    "TXN",
    "AVGO",
    "PYPL",
    "TSLA",
    "INTU",
    "SBUX",
    "BKNG",
    "GILD",
    "MDLZ",
    "ADP",
    "ISRG",
    "REGN",
    "QCOM",
    "VRTX",
    "ATVI",
]


for ticker in tickers:
    try:
        os.rename(f"./data/Stocks/{ticker.lower()}.us.txt", f"./data/ohlc/{ticker}.csv")
    except FileNotFoundError:
        print(f"File ./data/Stocks/{ticker.lower()}.us.txt not found for {ticker}")
