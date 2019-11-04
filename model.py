import math
import numpy as np
import pandas as pd
from pandas.core.series import Series as PandasSeries
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn import linear_model
from typing import NamedTuple, Dict, Optional, List
from enum import Enum
from structs import (
    DataPoint,
    ConfidenceBand,
    Seasonality,
    SeasonalityStatus,
    MetricStatus,
    Anomaly,
    AnomalyDirection,
)
from config import Config
from time_series import TimeSeries


class OnlineTSModel:
    """ Charisristic of the TimeSeries
    """
    def __init__(self) -> None:
        self.point: DataPoint = None
        self.count: int = 0
        # exponential moving average over the positive values
        self.positive_avg: float = 0.0
        self.lower: Optional[float] = None
        self.upper: Optional[float] = None

    def process(self, config: Config, point: DataPoint) -> None:
        self.point = point
        self.count += 1
        if point.value >= 0:
            alpha = config.exponential_baseline_alpha
            self.positive_avg = alpha * point.value + (1 - alpha) * self.positive_avg

        if not self.lower:
            self.lower = point.value
        else:
            self.lower = min(self.lower, point.value)
        if not self.upper:
            self.upper = point.value
        else:
            self.upper = max(self.upper, point.value)


class BaselineState(Enum):
    NA = 'na'
    INITIALIZING = 'initializing'
    RUNNING = 'running'


class BaselineModel:
    pass


class OfflineBaselineModel(BaselineModel):
    pass


class AdditiveHWOfflineModel(OfflineBaselineModel):

    def __init__(self, slen: int) -> None:
        self.slen = slen
        self.initial_trend = None
        self.initial_seasonals = None
        self.alpha = None
        self.beta = None
        self.gamma = None

    def get_initial_trend(self, config: Config, ts: PandasSeries) -> float:
        sum = 0.0
        for i in range(self.slen):
            sum += float(ts[i + self.slen] - ts[i]) / self.slen
        return sum / self.slen

    def get_initial_seasonal_components(self, config: Config, ts: PandasSeries) -> float:
        seasonals = {}
        season_averages = []
        n_seasons = int(len(ts) / self.slen)
        # compute season averages
        for j in range(n_seasons):
            season_averages.append(
                sum(ts[self.slen * j:self.slen * j + self.slen]) / float(self.slen)
            )
        # compute initial values
        for i in range(self.slen):
            sum_of_vals_over_avg = 0.0
            for j in range(n_seasons):
                sum_of_vals_over_avg += ts[self.slen * j + i] - season_averages[j]
            seasonals[i] = sum_of_vals_over_avg / n_seasons
        return seasonals

    def fit(self, config: Config, ts: TimeSeries) -> None:
        series = ts.df['value']
        self.initial_trend = self.get_initial_trend(config, series)
        self.initial_seasonals = self.get_initial_seasonal_components(config, series)
        # hack: using statsmodels just for finding the best params
        hw = ExponentialSmoothing(
            series, trend='add', damped=False, seasonal='add', seasonal_periods=self.slen
        ).fit()
        self.alpha, self.beta, self.gamma = (
            hw.params['smoothing_level'],
            hw.params['smoothing_slope'],
            hw.params['smoothing_seasonal'],
        )
        if config.debug_mode:
            param = {
                'alpha': self.alpha,
                'beta': self.beta,
                'gamma': self.gamma,
            }
            print(f'AdditiveHWOfflineModel.fit: {param}')


class OnlineBaselineModel(BaselineModel):

    def __init__(self) -> None:
        self.state = BaselineState.NA
        self.init_counter = None

    def initialize(self, config: Config, point: DataPoint) -> None:
        raise NotImplementedError

    def process(self, config: Config, point: DataPoint) -> None:
        self.update_state(config)
        if self.state == BaselineState.INITIALIZING:
            self.initialize(config, point)
            self.init_counter -= 1
        else:
            self.process_internal(config, point)

    def get_predicted_cb(self, config: Config) -> ConfidenceBand:
        if self.state == BaselineState.INITIALIZING:
            return ConfidenceBand(None, None)
        return self.get_cb_internal(config)

    def get_cb_internal(self, config: Config) -> ConfidenceBand:
        raise NotImplementedError

    def process_internal(self, config: Config, point: DataPoint) -> None:
        raise NotImplementedError

    def get_init_window(self, config: Config) -> int:
        raise NotImplementedError

    def update_state(self, config: Config) -> None:
        if self.state == BaselineState.NA:
            init_window = self.get_init_window(config)
            if init_window > 0:
                self.state = BaselineState.INITIALIZING
                self.init_counter = init_window
            else:
                self.state = BaselineState.RUNNING
        elif self.state == BaselineState.INITIALIZING:
            finished_init: bool = self.init_counter == 0
            if finished_init:
                self.state = BaselineState.RUNNING

    def states_for_debugging(self) -> Dict:
        return {}

    def stop_adaptation(self, config: Config) -> None:
        raise NotImplementedError

    def reset_adaptation(self, config: Config) -> None:
        raise NotImplementedError


class NormalDistributionOnlineModel(OnlineBaselineModel):

    def __init__(self, overriden_alpha: Optional[float] = None) -> None:
        super(NormalDistributionOnlineModel, self).__init__()
        self.sos: float = 0.0  # sum of sqaure
        self.avg: float = 0.0
        self.count: int = 0
        self.overriden_alpha: Optional[float] = overriden_alpha
        self.adaptation_factor: float = 1.0

    def initialize(self, config: Config, point: DataPoint) -> None:
        self.count += 1
        self.sos += (point.value * point.value - self.sos) / self.count
        self.avg += (point.value - self.avg) / self.count

    def process_internal(self, config: Config, point: DataPoint) -> None:
        alpha = config.normal_distribution_baseline_alpha
        if self.overriden_alpha:
            alpha = self.overriden_alpha
        alpha = alpha / self.adaptation_factor
        self.sos = (1 - alpha) * self.sos + alpha * point.value * point.value
        self.avg = (1 - alpha) * self.avg + alpha * point.value

    def get_cb_internal(self, config: Config, cbf: float = None) -> ConfidenceBand:
        cbf = cbf or config.normal_distribution_confidence_band_factor
        sigma = math.sqrt(abs(self.sos - self.avg * self.avg))
        return ConfidenceBand(
            lower=self.avg - cbf * sigma,
            upper=self.avg + cbf * sigma,
        )

    def get_init_window(self, config: Config) -> int:
        return config.normal_distribution_init_window

    def stop_adaptation(self, config: Config) -> None:
        self.adaptation_factor = config.normal_distribution_adaptation_factor

    def reset_adaptation(self, config: Config) -> None:
        self.adaptation_factor = 1.0


class AdditiveHWOnlineModel(OnlineBaselineModel):

    def __init__(
        self,
        initial_trend: float,
        seasonals: Dict[int, float],
        alpha: float,
        beta: float,
        gamma: float,
        slen: int,  # season length
        error_distr_alpha: float,
    ) -> None:
        super(AdditiveHWOnlineModel, self).__init__()
        # initial values
        self.initial_trend: float = initial_trend
        self.seasonals: Dict[int, float] = seasonals
        self.alpha: float = alpha
        self.beta: float = beta
        self.gamma: float = gamma
        self.slen: int = slen

        # internal param states
        self.i = 0
        self.last_smooth: float = None
        self.smooth: float = None
        self.trend: float = None
        self.pred: float = None
        self.resid: float = None  # the error
        self.error_distr: NormalDistributionOnlineModel = NormalDistributionOnlineModel(
            overriden_alpha=error_distr_alpha
        )
        self.adaptation_factor = 1.0

    @classmethod
    def from_offline_model(cls, offline_model: AdditiveHWOfflineModel, config: Config):
        return cls(
            initial_trend=offline_model.initial_trend,
            seasonals=offline_model.initial_seasonals,
            alpha=offline_model.alpha,
            beta=offline_model.beta,
            gamma=offline_model.gamma,
            slen=offline_model.slen,
            error_distr_alpha=config.additive_hw_error_norm_distr_alpha,
        )

    def initialize(self, config: Config, point: DataPoint) -> None:
        self.process_internal(config, point)

    def process_internal(self, config: Config, point: DataPoint) -> None:
        self.error_distr.state = self.state
        if self.i == 0:
            self.smooth = point.value
            self.trend = self.initial_trend
            self.pred = point.value
        else:
            alpha, beta, gamma = (
                self.alpha / self.adaptation_factor,
                self.beta / self.adaptation_factor,
                self.gamma / self.adaptation_factor,
            )
            val = point.value
            self.last_smooth, self.smooth = (
                self.smooth,
                alpha * (val - self.seasonals[self.i % self.slen]) +
                (1 - alpha) * (self.smooth + self.trend)
            )
            self.trend = beta * (self.smooth - self.last_smooth) + \
                (1 - beta) * self.trend
            self.seasonals[self.i % self.slen] = \
                gamma * (val - self.smooth) + \
                (1 - gamma) * self.seasonals[self.i % self.slen]
            self.pred = self.smooth + self.trend + self.seasonals[self.i % self.slen]
            self.resid = val - self.pred
            if self.state == BaselineState.INITIALIZING:
                self.error_distr.initialize(
                    config,
                    DataPoint(point.ts, self.resid),
                )
            else:
                self.error_distr.process_internal(
                    config,
                    DataPoint(point.ts, self.resid),
                )
        self.i += 1

    def get_cb_internal(self, config: Config) -> ConfidenceBand:
        cbf = config.additive_hw_confidence_band_factor
        error_cb = self.error_distr.get_cb_internal(config, cbf)
        pred_next = (self.smooth + self.trend) + self.seasonals[self.i % self.slen]
        return ConfidenceBand(
            lower=pred_next + error_cb.lower,
            upper=pred_next + error_cb.upper,
        )

    def get_init_window(self, config: Config) -> int:
        return config.normal_distribution_init_window

    def stop_adaptation(self, config: Config) -> None:
        # Could customize adaptation logic for this model specifically
        self.adaptation_factor = config.additive_hw_adaptation_factor
        self.error_distr.stop_adaptation(config)

    def reset_adaptation(self, config: Config) -> None:
        self.adaptation_factor = 1.0
        self.error_distr.reset_adaptation(config)

    def states_for_debugging(self) -> Dict:
        return {
            'smooth': self.smooth,
            'trend': self.trend,
            'pred': self.pred,
            'resid': self.resid,
        }


class OfflineMetricModel(NamedTuple):
    baseline_model: OnlineBaselineModel
    seasonality: Seasonality
    ts_model: OnlineTSModel


class AnomalyModel:

    def __init__(
        self,
        status: MetricStatus = MetricStatus.NA,
        tracked_normal_points: int = 0,  # actually track this
        anomaly: Anomaly = None,
    ) -> None:
        self.status = status
        self.tracked_normal_points = tracked_normal_points
        self.anomaly = anomaly

    def process(self, config: Config, point: DataPoint, cb: ConfidenceBand) -> None:
        above_upper = point.value > cb.upper
        below_lower = point.value < cb.lower
        breachCB = above_upper or below_lower

        if breachCB:
            self.status = MetricStatus.ABNORMAL
            direction = AnomalyDirection.UP if above_upper else AnomalyDirection.DOWN
            if self.anomaly:
                self.anomaly.direction = direction
                self.anomaly.pt_count += 1
            else:
                self.anomaly = Anomaly(
                    start_ts=point.ts,
                    end_ts=None,
                    direction=direction,
                )
        else:
            self.status = MetricStatus.NORMAL
            if self.anomaly:
                self.anomaly = None


class OnlineMetricModel:

    def __init__(
        self,
        baseline_model: OnlineBaselineModel = None,
        ts_model: OnlineTSModel = None,
        anomaly_model: AnomalyModel = None,
        has_been_updated_with_offline: bool = False,
    ) -> None:
        # TODO: use an algorithm provider to choose baseline
        self.baseline_model = baseline_model or NormalDistributionOnlineModel()
        self.ts_model = ts_model or OnlineTSModel()
        self.anomaly_model = anomaly_model or AnomalyModel()
        self.has_been_updated_with_offline = has_been_updated_with_offline

    @classmethod
    def from_offline_model(cls, offline_model: OfflineMetricModel):
        return cls(
            baseline_model=offline_model.baseline_model,
            ts_model=offline_model.ts_model,
        )

    def should_update(self):
        # TODO: use version and timestamp to keep track of models and decide
        # if we should update
        return not self.has_been_updated_with_offline


class InMemoryModelProvider:

    def __init__(self) -> None:
        self.online_model: Dict[str, OnlineMetricModel] = {}
        self.offline_model: Dict[str, OfflineMetricModel] = {}

    def load_and_update_online_model(self, key: str) -> Optional[OnlineMetricModel]:
        if key in self.offline_model:
            # try updating online model with offline model
            if (
                (key in self.online_model and self.online_model[key].should_update())
                or (key not in self.online_model)
            ):
                self.online_model[key] = OnlineMetricModel.from_offline_model(
                    self.offline_model[key]
                )
                self.online_model[key].has_been_updated_with_offline = True

        return self.online_model.get(key)

    def store_online_model(
        self, key: str, online_model: OnlineMetricModel
    ) -> None:
        self.online_model[key] = online_model

    def store_offline_model(
        self, key: str, offline_model: OfflineMetricModel
    ) -> None:
        self.offline_model[key] = offline_model


class Detrender:
    """ detrend using linear regression """

    def process(self, x: PandasSeries) -> PandasSeries:
        nobs = len(x)
        t = self.get_trend(np.array(range(nobs)).reshape(-1, 1), np.array(x.array))
        trend = pd.Series(t, index=x.index)
        detrended = x - trend
        return detrended

    def get_trend(self, x, y):
        reg = linear_model.LinearRegression().fit(x, y)
        return reg.predict(x)


class Correlogram(NamedTuple):
    lag: int
    corr: float


class SeasonalityModeler:

    def process(self, config: Config, ts: TimeSeries) -> Seasonality:
        # TODO: sampling detection in case input is not regular

        # TODO: better missing points hanlding and extract regular series
        ts.df = ts.df.fillna(ts.df.mean())

        detrended = Detrender().process(ts.df['value'])

        # finding first peak in segments,
        # consider using clustering delta for better results
        acf_data, conf = acf(
            detrended,
            nlags=min(ts.df.size - 1, config.seasonality_acf_max_lag),
            alpha=config.seasonality_acf_conf_interval_alpha,
            fft=False,
        )
        maximums = self.get_local_maximums(acf_data, conf)

        # TODO: round season period
        season = self.get_season_first_peak(maximums)
        if config.debug_mode:
            print(f'Season period: {season.period}, status: {season.seasonality_status}')

        return season

    def get_season_first_peak(self, maximums: List[Correlogram]) -> Seasonality:
        if maximums:
            return Seasonality(
                period=maximums[0].lag,
                sampling_period=0,  # TODO: fill this
                seasonality_status=SeasonalityStatus.SUCCESS_SEASON,
            )
        return Seasonality(0, 0, SeasonalityStatus.NO_SEASON)

    def get_local_maximums(
        self, acf_data: np.ndarray, conf: np.ndarray
    ) -> List[Correlogram]:
        upper_bound = conf[:, 1] - acf_data
        res = []
        in_segment = False
        max_corr = None
        max_period = None
        prev_u, prev_x = upper_bound[0], acf_data[0]

        lag = 1
        for u, x in zip(upper_bound[1:], acf_data[1:]):
            if in_segment:
                if x > max_corr:
                    max_period = lag
                    max_corr = x
            if prev_x < prev_u and x > u:
                # beginning of segment
                in_segment = True
                max_period = lag
                max_corr = x
            elif prev_x > prev_u and x < u:
                # end of segment
                in_segment = False
                if max_period:
                    res.append(Correlogram(max_period, max_corr))
                max_period = None
                max_corr = None
            lag += 1
            prev_u = u
            prev_x = x

        return res
