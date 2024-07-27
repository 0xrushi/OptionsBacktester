import gzip
import logging
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz
import yfinance as yf
from colorama import Fore, init
from dateutil.relativedelta import relativedelta
from typing import List, Tuple, Dict

# Initialize colorama for colored console output
init()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_yfinance_price(row: pd.Series) -> float | None:
    """
    Retrieve the closing price for a specific stock and date using yfinance.

    Args:
        row (pd.Series): A row from the DataFrame containing 'underlying' and 'quotedate'.

    Returns:
        float or None: The closing price for the specified date if found, otherwise None.
    """
    ticker = row['underlying']
    date = row['quotedate']

    # Input validation
    if not isinstance(ticker, str):
        logger.error(f"ticker must be a string, got {type(ticker)}")
        return None
    if not isinstance(date, (datetime, pd.Timestamp)):
        logger.error(f"date must be a datetime or Timestamp, got {type(date)}")
        return None

    # Create a Ticker object and download historical data
    stock = yf.Ticker(ticker)
    hist = stock.history(start=date, end=date + timedelta(days=1))

    if not hist.empty:
        return hist['Close'].iloc[0]
    else:
        logger.warning(f"No data found for {ticker} on {date}")
        return None


def third_friday(year: int, month: int) -> datetime:
    """
    Calculate the third Friday of a given month and year.

    Args:
        year (int): The year.
        month (int): The month.

    Returns:
        datetime: The third Friday of the specified month and year.
    """
    res = datetime(year, month, 15, tzinfo=pytz.utc)
    w = res.weekday()
    if w != 4:
        res = res.replace(day=15 + (4 - w) % 7)
    return res


class OptAlpha:
    """Base class for option alpha strategies."""

    def __init__(self, instruments: List[str], trade_range: Tuple[datetime, datetime], dfs: Dict[str, pd.DataFrame]):
        self.instruments = instruments
        self.trade_range = trade_range
        self.dfs = dfs
        self.data_buffer = []
        self.data_buffer_idx = []

    def archive_constructor(self, dt: datetime) -> str:
        return f"bb_{dt.year}_{dt.strftime('%B')}.zip"

    def filename_constructor(self, dt: datetime) -> str:
        return f"bb_options_{str(dt.date()).replace('-', '')}.csv"


class OptAlpha2(OptAlpha):
    """Extended class for option alpha strategies with additional functionality."""

    def __init__(self, instruments: List[str], trade_range: Tuple[datetime, datetime], dfs: Dict[str, pd.DataFrame]):
        super().__init__(instruments, trade_range, dfs)
        self.instantiate_variables()

    def instantiate_variables(self):
        """Initialize class variables."""
        self.loaded = set()
        self.data_buffer = []
        self.data_buffer_idx = []

    def archive_constructor(self, dt: datetime) -> str:
        """Construct archive filename."""
        return f"{dt.year}-{dt.month:02d}-{dt.day:02d}.csv.gz"

    def filename_constructor(self, dt: datetime):
        """Construct filename."""
        return f"{dt.year}-{dt.month:02d}-{dt.day:02d}.csv"

    @staticmethod
    def screen_universe(df: pd.DataFrame, universe: List[str]) -> pd.DataFrame:
        """
        Screen and process the options data.

        Args:
            df (pd.DataFrame): Raw options data.
            universe (list): List of instruments to include.

        Returns:
            pd.DataFrame: Processed options data.
        """
        logger.info("Processing options data...")

        # Process the dataframe
        df['split_text'] = df['ticker'].apply(lambda x: re.split(r'(?<=C|P)(?!.*[CP])', x))
        df['strike'] = df['split_text'].apply(lambda x: float(x[-1]) / 1000)
        df['type'] = df['split_text'].apply(lambda x: "call" if x[0][-1] == "C" else "put")
        df['underlying'] = df['split_text'].apply(lambda x: x[0][:-1][2:-6])
        df['last'] = df['close']
        df['OptionRoot'] = df['ticker']
        df['openinterest'] = 0.0  # not available in data

        # Convert dates
        df['expiration'] = pd.to_datetime(df['split_text'].apply(lambda x: x[0][:-1][-6:]), format='%y%m%d')
        df['quotedate'] = pd.to_datetime(df['window_start'])
        df['window_start'] = pd.to_datetime(df['window_start'])
        df["dte"] = (df.expiration - df.quotedate).apply(lambda x: x.days)

        # Get underlying price
        df['underlying_last'] = df.apply(get_yfinance_price, axis=1)

        # Filter and process data
        df = df.loc[df.volume != 0]
        df["in_universe"] = df.underlying.apply(lambda x: x in universe)
        df = df.loc[df.in_universe].drop(columns=["in_universe"])

        # Localize timezone
        df.expiration = pd.to_datetime(df.expiration).dt.tz_localize("UTC")
        df.quotedate = pd.to_datetime(df.quotedate).dt.tz_localize("UTC")

        # Drop unnecessary columns
        df.drop(['split_text', 'ticker', 'open', 'high', 'low', 'close', 'volume'], axis=1, inplace=True)

        # Rename columns
        df = df.rename(columns={
            "OptionRoot": "optionroot",
            "underlying": "underlying",
            "underlying_last": "underlying_last",
            "type": "type",
            "expiration": "expiration",
            "quotedate": "quotedate",
            "strike": "strike",
            "last": "last",
            "openinterest": "openinterest",
            "volume": "volume"
        })

        logger.info("Options data processed successfully.")
        return df.set_index("optionroot", drop=True)

    def load_buffer(self, load_from, test_end, min_buffer_len=100, min_hist_len=2):
        """
        Load data into the buffer.

        Args:
            load_from (datetime): Start date for loading data.
            test_end (datetime): End date for loading data.
            min_buffer_len (int): Minimum buffer length.
            min_hist_len (int): Minimum historical data length.
        """
        logger.info("Loading data buffer...")

        _dir = "polygon/options/options_trades/day_aggs_temp/"

        if any(dt >= load_from for dt in self.data_buffer_idx):
            logger.info(f"Data already loaded from {load_from}. Skipping.")
            return

        self.data_buffer = self.data_buffer[-min_hist_len:]
        self.data_buffer_idx = self.data_buffer_idx[-min_hist_len:]

        while len(self.data_buffer) < min_buffer_len:
            while self.archive_constructor(dt=load_from) in self.loaded:
                load_from += relativedelta(days=1)
            if load_from > test_end:
                break

            an = self.archive_constructor(dt=load_from)
            file_path = os.path.join(_dir, an)
            pat = os.path.basename(file_path).replace(".csv.gz", "")

            if os.path.exists(f'data/optdat_{pat}.parquet'):
                self._load_parquet_file(pat, an)
            elif os.path.exists(file_path):
                self._load_csv_file(file_path, pat, an)
            else:
                logger.error(f"File not found: {file_path}")

        self.compute_buffer()
        logger.info("Data buffer loaded successfully.")

    def _load_parquet_file(self, pat, an):
        """Load data from a parquet file."""
        optdat = pd.read_parquet(f'data/optdat_{pat}.parquet')
        self._process_loaded_data(optdat, an)

    def _load_csv_file(self, file_path, pat, an):
        """Load data from a CSV file."""
        with gzip.open(file_path, 'rt') as f:
            optdat = pd.read_csv(f)
            optdat = self.screen_universe(df=optdat, universe=self.instruments)
            optdat.to_parquet(f'data/optdat_{pat}.parquet')
            self._process_loaded_data(optdat, an)

    def _process_loaded_data(self, optdat, an):
        """Process loaded data and add to buffer."""
        self.data_buffer.append(optdat)
        yyyymmdd = an.split(".csv")[0]
        self.data_buffer_idx.append(
            datetime(
                year=int(yyyymmdd[:4]),
                month=int(yyyymmdd[5:7]),
                day=int(yyyymmdd[8:10]),
                tzinfo=pytz.utc
            )
        )
        self.loaded.add(an)

    def compute_buffer(self):
        """Compute strategy buffer from loaded data."""
        logger.info("Computing strategy buffer...")
        strat_buffer = []
        for optdat, optidx in zip(self.data_buffer, self.data_buffer_idx):
            data = optdat.copy()
            next_month = optidx + relativedelta(months=1)
            second_monthlies = third_friday(year=next_month.year, month=next_month.month)
            thurs_fri_sat = set(
                [second_monthlies - relativedelta(days=1), second_monthlies, second_monthlies + relativedelta(days=1)])

            data["strike_dist"] = np.abs(data.underlying_last - data.strike)
            dat_insts = set(data["underlying"])
            temp = {}
            for inst in sorted(dat_insts):
                inst_dat = data.loc[data.underlying == inst]
                calls = inst_dat[inst_dat['type'] == 'call']
                puts = inst_dat[inst_dat['type'] == 'put']

                min_call = calls.loc[calls['strike_dist'] == calls['strike_dist'].min()]
                min_call = min_call.loc[min_call.strike == min_call.strike.min()]

                min_put = puts.loc[puts['strike_dist'] == puts['strike_dist'].min()]
                min_put = min_put.loc[min_put.strike == min_put.strike.min()]

                inst_dat = pd.concat([min_call, min_put])
                temp.update(inst_dat.to_dict("index"))

            atm_df = pd.DataFrame.from_dict(temp, orient="index")
            strat_buffer.append(atm_df)

        self.strat_buffer = strat_buffer
        logger.info("Strategy buffer computed successfully.")

    def get_pnl(self, date, last):
        """
        Calculate the profit and loss for a given date.

        Args:
            date (datetime): The date to calculate PnL for.
            last (dict): The last known positions.

        Returns:
            float: The calculated PnL.
        """
        logger.info(f"Calculating PnL for {date}...")

        try:
            cur_idx = self.data_buffer_idx.index(date)
        except ValueError:
            logger.error(f"Date {date} not found in data buffer")
            return 0.0

        if cur_idx == 0:
            logger.warning(f"No previous data available for date {date}")
            return 0.0

        curr = self.data_buffer[cur_idx]
        prev = self.data_buffer[cur_idx - 1]

        pnl_list = []
        for ticker, positions in last.items():
            for option_type in ['C', 'P']:
                for option, unit in zip(positions[option_type], positions[f"{option_type}U"]):
                    curr_price = curr.at[option, "last"] if option in curr.index else None
                    prev_price = prev.at[option, "last"] if option in prev.index else None
                    if curr_price is not None and prev_price is not None:
                        pricedelta = curr_price - prev_price
                        pnl_list.append(pricedelta * unit)
                    else:
                        logger.warning(f"Price data missing for {option_type} option {option}")

        total_pnl = sum(pnl_list)
        logger.info(f"PnL for {date}: {total_pnl}")
        return float(total_pnl)

    @staticmethod
    def _default_pos():
        """Create a default position dictionary."""
        return defaultdict(lambda: {"S": 0, "C": [], "P": [], "CU": [], "PU": []})

    def compute_signals(self, date, capital):
        """
        Compute trading signals for a given date and capital.

        Args:
            date (datetime): The date to compute signals for.
            capital (float): The available capital.

        Returns:
            dict: The computed trading signals.
        """
        logger.info(f"Computing signals for {date}...")

        if date not in self.data_buffer_idx:
            logger.warning(f"No data available for {date}")
            return None

        date_data = self.strat_buffer[self.data_buffer_idx.index(date)]
        trade_insts = set(date_data["underlying"])
        underlying = {inst: date_data.loc[date_data.underlying.values == inst].underlying_last.values[0] for inst in
                      trade_insts}

        notional_leverage = 3
        notional_per_trade = capital * notional_leverage / len(trade_insts)
        signal_dict = self._default_pos()

        for inst in trade_insts:
            pos = notional_per_trade / underlying[inst] * -1
            signal_dict[inst] = {
                "S": 0,
                "C": [date_data.loc[np.logical_and(date_data.underlying.values == inst,
                                                   date_data.type.values == "call")].index.values[0]],
                "P": [date_data.loc[np.logical_and(date_data.underlying.values == inst,
                                                   date_data.type.values == "put")].index.values[0]],
                "CU": [pos],
                "PU": [pos],
            }

        logger.info(f"Signals computed successfully for {date}")
        return signal_dict

    async def run_simulation(self):
        trade_start = self.trade_range[0]
        trade_end = self.trade_range[1]

        trade_range = pd.date_range(
            start=datetime(trade_start.year, trade_start.month, trade_start.day),
            end=datetime(trade_end.year, trade_end.month, trade_end.day),
            freq="D",
            tz=pytz.utc
        )

        print(trade_range)
        breakpoint()

        portfolio_df = pd.DataFrame(index=trade_range).reset_index().rename(columns={"index": "datetime"})
        portfolio_df.at[0, "capital"] = 10000.0

        self.data_buffer = []
        self.data_buffer_idx = []
        self.loaded = set()

        # breakpoint()
        last_positions = self._default_pos()
        for i in portfolio_df.index:
            date = portfolio_df.at[i, "datetime"]
            self.load_buffer(load_from=date, test_end=trade_end, min_buffer_len=180, min_hist_len=2)

            print(f"{Fore.RED}self databuffer is {Fore.RESET}{self.data_buffer}")
            print(f"{Fore.RED}strat_buffer is {Fore.RESET}{self.strat_buffer}")
            print(f"{Fore.RED}i is {i}{Fore.RESET}")
            print(f"{Fore.RED}load_from is {date}{Fore.RESET}")
            print(f"{Fore.RED}trade_end is {trade_end}{Fore.RESET}")
            breakpoint()
            if i != 0:
                day_pnl = self.get_pnl(date=date, last=last_positions)
                print(day_pnl)
                # breakpoint()
                previous_capital = portfolio_df.at[i - 1, 'capital']
                new_capital = float(previous_capital + day_pnl)  # Ensure it's a float
                portfolio_df.at[i, 'capital'] = new_capital

                current_capital = portfolio_df.at[i, 'capital']
                signal_dict = self.compute_signals(date=date, capital=current_capital)
                last_positions = signal_dict if signal_dict else last_positions

                if i % 20 == 0:
                    print(f"Capital at step {i}: {current_capital}")

        return portfolio_df


def sp500_constituents():
    url = 'https://en.m.wikipedia.org/wiki/List_of_S%26P_500_companies'
    df = pd.read_html(url, attrs={'id': 'constituents'}, index_col='Symbol')[0]
    return list(df.index)


async def main():
    trade_start = datetime(2024, 6, 10, tzinfo=pytz.utc)
    trade_end = datetime(2024, 6, 13, tzinfo=pytz.utc)
    insts = ["MMM"]

    print(trade_start)

    strat = OptAlpha2(
        instruments=insts,
        trade_range=(trade_start, trade_end),
        dfs={}
    )

    try:
        df = await strat.run_simulation()
        print(df)
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

    # all(self.data_buffer[0]==self.data_buffer[2])
