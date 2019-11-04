import numpy as np
import warnings
from typing import Optional, List
from model import (
    AdditiveHWOfflineModel,
    AdditiveHWOnlineModel,
    InMemoryModelProvider,
    OnlineTSModel,
    OnlineMetricModel,
    OfflineMetricModel,
    SeasonalityModeler,
    BaselineState,
)
from structs import DataPoint, AnomalyResult, MetricStatus
from config import Config
from time_series import TimeSeries


class BatchMetricProcessor:

    def process_batch(
        self, key: str, config: Config, ts: TimeSeries
    ) -> List[AnomalyResult]:
        # 1. call calculate_offline_model to get offline model
        # 2. process the data points again to get anomaly
        pass

    def calculate_offline_model(
        self, key: str, config: Config, ts: TimeSeries
    ) -> OfflineMetricModel:
        seasonality = SeasonalityModeler().process(config, ts)
        offline_model = AdditiveHWOfflineModel(seasonality.period)
        offline_model.fit(config, ts)

        online_model = AdditiveHWOnlineModel.from_offline_model(
            offline_model, config
        )
        ts_model = OnlineTSModel()
        for pt in ts.stream_data_points():
            online_model.process(config, pt)
            ts_model.process(config, pt)

        # TODO: unhack this
        if config.additive_hw_overriding_online_params:
            online_model.alpha, online_model.beta, online_model.gamma = (
                config.additive_hw_online_alpha,
                config.additive_hw_online_beta,
                config.additive_hw_online_gamma,
            )

        return OfflineMetricModel(
            baseline_model=online_model,
            seasonality=seasonality,
            ts_model=ts_model,
        )


class StreamMetricProcessor:

    def __init__(self, model_provider: InMemoryModelProvider) -> None:
        self.model_provider = model_provider

    def process(
        self, key: str, config: Config, point: DataPoint
    ) -> Optional[AnomalyResult]:
        if not point or np.isnan(point.value):
            warnings.warn(f"Invalid point at timestamp {point.ts}!", RuntimeWarning)
            return

        m = self.model_provider.load_and_update_online_model(key)
        if not m:
            m = OnlineMetricModel()

        m.ts_model.process(config, point)
        cb = m.baseline_model.get_predicted_cb(config)
        if m.baseline_model.state == BaselineState.RUNNING:
            m.anomaly_model.process(config, point, cb)

        if m.anomaly_model.status == MetricStatus.ABNORMAL and \
                m.anomaly_model.anomaly.pt_count < config.adaptation_drop_count:
            m.baseline_model.stop_adaptation(config)
        else:
            m.baseline_model.reset_adaptation(config)

        m.baseline_model.process(config, point)

        # TODO: publish to anomaly stream

        self.model_provider.store_online_model(key, m)

        return AnomalyResult(
            point,
            cb,
            m.anomaly_model.anomaly,
            debug_info=m.baseline_model.states_for_debugging()
            if config.debug_mode else {}
        )
