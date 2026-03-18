import pandas as pd
import yfinance as yf


def download_sp500_data(start_date: str, end_date: str, ticker: str = "^GSPC") -> pd.DataFrame:
    """Download S&P 500 (or any index) daily close prices and compute log returns.

    Parameters
    ----------
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.
    ticker : str, optional
        Ticker symbol to download. Defaults to '^GSPC' (S&P 500).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``close`` and ``returns`` indexed by date.
    """
    raw = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")

    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.rename("close")

    returns = close.pct_change().dropna().rename("returns")

    data = pd.concat([close, returns], axis=1).dropna()
    return data
