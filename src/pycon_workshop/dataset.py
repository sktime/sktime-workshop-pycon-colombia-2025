from sktime.datasets.forecasting._base import BaseForecastingDataset
import numpy as np
import pandas as pd
from sktime.split.temporal_train_test_split import temporal_train_test_split


class PyConWorkshopDataset(BaseForecastingDataset):

    _tags = {
        "object_type": ["dataset_forecasting", "dataset"],
        "task_type": ["forecaster"],
        # Estimator type
        "is_univariate": True,
        "is_equally_spaced": True,
        "has_nans": False,
        "has_exogenous": True,
        "n_instances": 50,
        "n_instances_train": 50,
        "n_instances_test": 50,
        "n_timepoints": 455 + 90,
        "n_timepoints_train": 455,
        "n_timepoints_test": 90,
        "frequency": "D",
        "n_dimensions": 1,
        "is_one_panel": False,
        "n_panels": 1,
        "n_hierarchy_levels": 0,
        "is_one_series": False,
        "n_splits": 1,
    }

    def __init__(self, mode="univariate"):

        self.mode = mode
        super().__init__()
        self._cached = False

        if mode == "panel":
            self.set_tags(
                **{
                    "is_one_panel": False,
                    "n_panels": 5,
                    "n_hierarchy_levels": 1,
                }
            )
        if mode == "hierarchical":
            self.set_tags(
                **{
                    "is_one_panel": False,
                    "n_panels": 50,
                    "n_hierarchy_levels": 2,
                }
            )

    def _cache_dataset(self):
        df = _generate_dataset()
        self._y = df[["sales"]]
        self._X = df[["promo", "macro_index"]]

        self._X_train, self._X_test, self._y_train, self._y_test = (
            temporal_train_test_split(self._X, self._y, test_size=180)
        )

        self.cache_ = {
            "X": self._X,
            "y": self._y,
            "X_train": self._X_train,
            "X_test": self._X_test,
            "y_train": self._y_train,
            "y_test": self._y_test,
        }

        self._cached = True

    def _load(self, *args):

        if not self._cached:
            self._cache_dataset()

        preprocessed_cache = self._preprocess_cached()
        outs = tuple([preprocessed_cache[key] for key in args])
        return outs

    def _preprocess_cached(self):

        groupby_keys = ["group_id", "sku_id"]

        if self.mode == "univariate":
            groupby_keys = []
        elif self.mode == "panel":
            groupby_keys = ["sku_id"]
        elif self.mode == "hierarchical":
            groupby_keys = ["group_id", "sku_id"]
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        _agg = {
            "X": {"promo": "mean", "macro_index": "mean"},
            "y": {"sales": "sum"},
        }

        cache = {
            key: df.groupby(groupby_keys + ["date"]).agg(_agg[key.split("_")[0]])
            for key, df in self.cache_.items()
        }
        return cache


def _generate_dataset(
    start="2020-01-01",
    end="2025-01-01",
    n_skus=50,
    seed=123,
    base_lambda=5,
    macro_trend_amp=5,
    promo_prob=0.05,
    promo_effect=1.5,
    # new / tuned knobs ↓
    trend_slope=0.01,  # upward drift per day
    season_amp=0.30,  # multiplicative seasonal amplitude (±30 %)
    zero_inf_start=0.70,  # 70 % zeros at the very beginning
    zero_inf_end=0.15,  # 15 % zeros at the very end
):
    """
    Synthetic SKU-level daily sales with:
      • deterministic upward trend
      • multiplicative seasonality
      • groups with positive / negative / no correlation
      • exogenous macro index & promo flags
      • time-varying zero-inflation (sparser early, denser later)
    """
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 1. Groups
    # ------------------------------------------------------------------
    positive_group = list(range(0, 5))
    negative_groups = [list(range(5, 10)), list(range(10, 15)), list(range(15, 20))]

    # ------------------------------------------------------------------
    # 2. Calendar & deterministic components
    # ------------------------------------------------------------------
    dates = pd.period_range(start, end, freq="D")
    n_days = len(dates)

    day_of_year = dates.dayofyear.values
    yearly = np.sin(2 * np.pi * day_of_year / 365.25)
    weekly = np.where(dates.weekday < 5, 1.0, 1.3)

    seasonality_factor = 1.0 + season_amp * yearly * weekly  # multiplicative
    latent_trend = trend_slope * np.arange(n_days)  # linear ↑ trend

    window = 3

    # ------------------------------------------------------------------
    # 3. Exogenous macro index
    # ------------------------------------------------------------------
    macro_index = (
        100
        + macro_trend_amp * np.sin(2 * np.pi * day_of_year / 180)
        + rng.normal(0, 2, n_days)
    )
    macro_index_z = (macro_index - macro_index.mean()) / macro_index.std()

    # ------------------------------------------------------------------
    # 4. Time-varying zero-inflation probability
    # ------------------------------------------------------------------
    zero_prob_t = np.linspace(zero_inf_start, zero_inf_end, n_days)  # high ➜ low

    # ------------------------------------------------------------------
    # 5. Helper for row assembly
    # ------------------------------------------------------------------
    frames = []

    def make_rows(sku, sales, promo_flags, group_id):
        return pd.DataFrame(
            {
                "date": dates,
                "sku_id": sku,
                "sales": sales,
                "promo": promo_flags.astype(int),
                "macro_index": macro_index,
                "group_id": group_id,
            }
        )

    # ------------------------------------------------------------------
    # 6. Positively correlated SKUs
    # ------------------------------------------------------------------
    shared_noise_pos = rng.normal(0, 0.3, n_days)
    for sku in positive_group:
        gamma = rng.normal(0.7, 0.2)
        promo_flags = rng.random(n_days) < promo_prob
        mean = (
            base_lambda
            + latent_trend
            + shared_noise_pos
            + gamma * macro_index_z
            + promo_effect * promo_flags
        )
        lam = np.exp(mean / 5) * seasonality_factor
        sales = rng.poisson(lam)
        sales[rng.random(n_days) < zero_prob_t] = 0

        sales = centred_moving_average_with_edges(sales, window=window)
        frames.append(make_rows(sku, sales, promo_flags, group_id=0))

    # ------------------------------------------------------------------
    # 7. Negative-correlation groups
    # ------------------------------------------------------------------
    for g_idx, group in enumerate(negative_groups, start=1):
        k = len(group)
        gamma_group = rng.normal(0.5, 0.3)
        group_noise = rng.normal(0, 0.4, n_days)
        group_mean = (
            base_lambda + latent_trend + group_noise + gamma_group * macro_index_z
        )
        group_lambda = np.exp(group_mean / 5) * seasonality_factor
        group_sales_total = rng.poisson(group_lambda)

        alpha = np.ones(k) * 0.8
        base_probs = rng.dirichlet(alpha, n_days)  # (n_days, k)
        promo_mat = rng.random((n_days, k)) < promo_prob
        adj_probs = base_probs * (1 + 0.3 * promo_mat)
        adj_probs = adj_probs / adj_probs.sum(axis=1, keepdims=True)

        sales_mat = np.vstack(
            [
                rng.multinomial(int(group_sales_total[d]), adj_probs[d])
                for d in range(n_days)
            ]
        )

        zero_mask = rng.random((n_days, k)) < zero_prob_t[:, None]
        sales_mat[zero_mask] = 0

        sales_mat = np.apply_along_axis(
            lambda x: centred_moving_average_with_edges(x, window=window),
            axis=0,
            arr=sales_mat,
        )
        sales_mat = np.floor(sales_mat).astype(int)

        for j, sku in enumerate(group):
            frames.append(
                make_rows(sku, sales_mat[:, j], promo_mat[:, j], group_id=g_idx)
            )

    # ------------------------------------------------------------------
    # 8. Independent SKUs
    # ------------------------------------------------------------------
    independent_skus = [
        s
        for s in range(n_skus)
        if s not in positive_group and all(s not in g for g in negative_groups)
    ]

    for sku in independent_skus:
        gamma = rng.normal(0.0, 0.5)
        promo_flags = rng.random(n_days) < promo_prob
        mean = (
            base_lambda
            + latent_trend
            + rng.normal(0, 0.5, n_days)
            + gamma * macro_index_z
            + promo_effect * promo_flags
        )
        lam = np.exp(mean / 5) * seasonality_factor
        sales = rng.poisson(lam)
        sales[rng.random(n_days) < zero_prob_t] = 0

        sales = centred_moving_average_with_edges(sales, window=window)
        frames.append(make_rows(sku, sales, promo_flags, group_id=-1))

    # ------------------------------------------------------------------
    # 9. Pack and return
    # ------------------------------------------------------------------

    df = pd.concat(frames, ignore_index=True)

    df = df.sort_values(["sku_id", "date"]).set_index(["group_id", "sku_id", "date"])
    df = df.sort_index()
    return df


def centred_moving_average_with_edges(sales, window=14, *, dtype=int):
    """
    Centred moving average that pads with the first/last values so the output
    has the same length as the input and no edge shrinkage.

    Parameters
    ----------
    sales : array-like
        1-D sequence of numbers.
    window : int, optional (default=14)
        Size of the sliding window (must be ≥ 1).
    dtype : NumPy dtype, optional (default=int)
        Integer type for the final, floored result.

    Returns
    -------
    np.ndarray
        Smoothed series of the same length as `sales`, with initial padding
        replicated from the first element and final padding from the last.
    """
    sales = np.asarray(sales, dtype=float)
    if sales.ndim != 1:
        raise ValueError("`sales` must be a 1-D sequence")
    if window < 1:
        raise ValueError("`window` must be at least 1")

    # --- 1. pad with first/last values so every centred window is complete
    pad_left = window // 2  # floor((window-1)/2)
    pad_right = window - 1 - pad_left  # ceil((window-1)/2)
    sales_padded = np.pad(
        sales, (pad_left, pad_right), mode="edge"
    )  # repeats edge values

    # --- 2. centred moving average
    weights = np.full(window, 1.0 / window, dtype=float)
    smoothed = np.convolve(sales_padded, weights, mode="valid")  # len == len(sales)

    # --- 3. floor to integer type if desired
    return np.floor(smoothed).astype(dtype)
