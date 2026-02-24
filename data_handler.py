import pandas as pd
import yfinance as yf


def download_data(tickers, start_date, end_date):
    """Download adjusted close prices for a list of tickers.

    Parameters
    ----------
    tickers : list of str
        Ticker symbols to download.
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.

    Returns
    -------
    pd.DataFrame
        DataFrame with adjusted close prices for each ticker, indexed by date.
    """
    frames = {}
    for ticker in tickers:
        raw = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if raw.empty:
            raise ValueError(f"No data returned for ticker '{ticker}'.")
        frames[ticker] = raw["Close"]

    prices = pd.DataFrame(frames)
    prices.dropna(how="all", inplace=True)
    prices.dropna(axis=1, how="all", inplace=True)
    return prices
