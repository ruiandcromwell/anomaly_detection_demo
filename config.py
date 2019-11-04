from typing import NamedTuple


class Config(NamedTuple):
    """ Based on the settings of 5m time scale
    TODO: generate config based on time scale
    """
    debug_mode: bool = False

    exponential_baseline_alpha: float = 0.05
    normal_distribution_baseline_alpha: float = 0.025
    normal_distribution_init_window: int = 20
    normal_distribution_confidence_band_factor: float = 5.0
    normal_distribution_adaptation_factor: float = 1000.0
    additive_hw_confidence_band_factor: float = 10.0
    additive_hw_error_norm_distr_alpha: float = 0.01
    additive_hw_overriding_online_params: bool = False
    additive_hw_online_alpha: float = 0.05
    additive_hw_online_beta: float = 0.005
    additive_hw_online_gamma: float = 0.05
    additive_hw_adaptation_factor: float = 10.0
    adaptation_drop_count: int = 10

    # seasonality
    seasonality_acf_max_lag: int = 500
    seasonality_acf_conf_interval_alpha: float = 0.05
