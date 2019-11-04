from enum import Enum
from pandas._libs.tslibs.timestamps import Timestamp
from typing import NamedTuple, Optional, Dict


class AnomalyDirection(Enum):
    UP = 'up'
    DOWN = 'down'


class Anomaly:

    def __init__(
        self,
        start_ts: Timestamp,
        end_ts: Optional[Timestamp],
        direction: AnomalyDirection,
        # TODO: implement below
        score: float = 0,
        magnitude: float = 0,
        pt_count: int = 0,
    ) -> None:
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.direction = direction
        self.score = score
        self.magnitude = magnitude
        self.pt_count = pt_count


class DataPoint(NamedTuple):
    ts: Timestamp
    value: float


class ConfidenceBand(NamedTuple):
    lower: Optional[float]
    upper: Optional[float]

    def is_defined(self) -> bool:
        return self.lower is not None and self.upper is not None


# BaselineResult
class AnomalyResult(NamedTuple):
    point: DataPoint
    cb: ConfidenceBand
    anomaly: Anomaly
    debug_info: Dict


class MetricStatus(Enum):
    NA = 'na'  # baseline not established yet
    NORMAL = 'normal'
    ABNORMAL = 'abnormal'
    TRACKED = 'tracked'  # metric within the baseline confidence band after an anomaly


class SeasonalityStatus(Enum):
    SUCCESS_SEASON = 'success_season'
    NO_SEASON = 'no_season'


class Seasonality(NamedTuple):

    period: int  # the number of data points that form a complete season
    sampling_period: int
    seasonality_status: SeasonalityStatus


class Sampling(NamedTuple):
    is_regular: bool
    # % of samples that meet the dominant sampling of the time series
    regularity: float
    sampling_period: int  # in seconds
