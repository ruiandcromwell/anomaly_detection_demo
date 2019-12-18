from typing import Generator
import pandas as pd
import matplotlib.pyplot as plt
from structs import DataPoint


class TimeSeries:

    def __init__(self, key: str, df: pd.DataFrame) -> None:
        self.key = key
        self.df = df

    @classmethod
    def fromCSV(cls, key: str, path: str):

        def convert_values(x):
            if '%' in x:
                return float(x.strip('%')) / 100
            else:
                return float(x)

        df = pd.read_csv(
            path,
            skiprows=1,
            names=['ts', 'value'],
            parse_dates=['ts'],
            converters={'value': convert_values},
            index_col='ts',
        )
        df.sort_index(inplace=True)
        return cls(key, df)

    def plot(self) -> None:
        plt.plot(self.df['value'])

    def stream_data_points(self) -> Generator[DataPoint, None, None]:
        for ts, row in self.df.iterrows():
            yield DataPoint(ts, row['value'])
