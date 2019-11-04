import matplotlib.pyplot as plt
from pandas._libs.tslibs.timestamps import Timestamp
from typing import List
from structs import AnomalyResult
from collections import defaultdict


def plot_anomaly(
    result: List[AnomalyResult],
    begin_ts: str = None,
    end_ts: str = None,
) -> None:
    if begin_ts and end_ts:
        result = [
            r
            for r in result
            if r.point.ts < Timestamp(end_ts) and
            r.point.ts > Timestamp(begin_ts)
        ]

    # original ts
    plt.plot([r.point.ts for r in result], [r.point.value for r in result])

    # confidence band
    x = [r.point.ts for r in result if r.cb.is_defined()]
    lower = [r.cb.lower for r in result if r.cb.is_defined()]
    upper = [r.cb.upper for r in result if r.cb.is_defined()]
    plt.fill_between(x, lower, upper, alpha=0.2)

    # anomaly
    prev_anomaly = None
    x = []
    y = []
    for r in result:
        if r.anomaly:
            if prev_anomaly:
                if prev_anomaly.start_ts != r.anomaly.start_ts:
                    plt.plot(x, y, 'r')
                    x = []
                    y = []
                    prev_anomaly = r.anomaly
            else:
                prev_anomaly = r.anomaly
            x.append(r.point.ts)
            y.append(r.point.value)

    plt.plot(x, y, 'r')

    # debugging by plotting internal model states by state name
    debug_info = defaultdict(list)
    for r in result:
        for k, v in r.debug_info.items():
            debug_info[k].append((r.point.ts, v))
    for k, values in debug_info.items():
        x = []
        y = []
        for ts, v in values:
            x.append(ts)
            y.append(v)
        plt.plot(x, y, label=k)
    plt.legend(loc='best')
