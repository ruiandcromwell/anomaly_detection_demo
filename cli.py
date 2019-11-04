import numpy as np
import matplotlib.pyplot as plt
from time_series import TimeSeries
from metric_processor import StreamMetricProcessor, BatchMetricProcessor
from model import InMemoryModelProvider
from config import Config
from utils import plot_anomaly


DATASET_PATH = '/Users/wangrui/Documents/anomaly/datasets/'


def demo_anomaly(dataset, config=None):
    if not config:
        config = Config()
    key = dataset
    ts = TimeSeries.fromCSV(dataset, f"{DATASET_PATH}{dataset}.csv")
    ts.plot()
    plt.show()

    model_provider = InMemoryModelProvider()
    processor = StreamMetricProcessor(model_provider)
    df1, df2 = np.array_split(ts.df, 2)
    ts1, ts2 = TimeSeries(ts.key, df1), TimeSeries(ts.key, df2)

    # step1: stream the first half
    res = []
    for pt in ts1.stream_data_points():
        r = processor.process(key, config, pt)
        if r:  # TODO: handle missing points and none better
            res.append(r)
    plot_anomaly(res)
    plt.show()

    # step2: process the first half offline and store the offline model
    bmp = BatchMetricProcessor()
    offline_model = bmp.calculate_offline_model(key, config, ts1)
    model_provider.store_offline_model(key, offline_model)

    # step3: stream the second half based on what we learned from first half
    for pt in ts2.stream_data_points():
        r = processor.process(key, config, pt)
        if r:  # TODO: handle missing points and none better
            res.append(r)

    return res


if __name__ == "__main__":
    result = demo_anomaly('daily_temperatures')
    plot_anomaly(result)
