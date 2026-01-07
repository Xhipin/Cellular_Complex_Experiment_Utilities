from ..registry.stat_registry import statistic, STATISTICS_REGISTRY
import numpy as np

@statistic()
def _standard_error(trials_data, ddof=1, axis=0):
    """
    Compute standard error of the mean for multiple trial curves.
    
    Parameters
    ----------
    trials_data : array-like or list of arrays
        Each element is a trial's values over time, or a 2D array with trials along `axis`.
    ddof : int
        Delta degrees of freedom for standard deviation.
    axis : int
        Axis along which to compute the standard deviation (trials axis).
    
    Returns
    -------
    np.ndarray
        Standard error per timestep.
    """
    import numpy as np

    if trials_data is None:
        raise ValueError("trials_data must be provided to compute standard error.")

    # Convert list of arrays to a 2D numpy array
    arr = np.array(trials_data, dtype=float)
    if arr.ndim != 2:
        raise ValueError("trials_data must be a 2D array or list of 1D arrays of equal length.")

    # Count non-NaN entries at each timestep to handle missing data
    n_eff = np.sum(~np.isnan(arr), axis=axis)
    std = np.nanstd(arr, axis=axis, ddof=ddof)
    se = std / np.sqrt(n_eff)
    return se


from types import MethodType

class StatisticsManager:
    def __init__(self, bootstrap=False):
        self._bootstrap = bool(bootstrap)

        # attach each registered statistic as an instance method
        def _make_stat_method(func):
            def _method(self, **kwargs):
                if getattr(self, "_bootstrap", False):
                    return self._bootstrapper(func, **kwargs)
                return func(**kwargs)
            return _method

        for key, func in STATISTICS_REGISTRY.items():
            public_name = key[1:] if isinstance(key, str) and key.startswith("_") else key
            setattr(self, public_name, MethodType(_make_stat_method(func), self))
    @staticmethod
    def _bootstrapper(func, **kwargs):
        """
        Generic bootstrapper for per-timestep uncertainty of trial curves.

        Parameters (passed via **kwargs)
        --------------------------------
        trials_data : array-like or list of 1D arrays
            Each element is a trial's values over time. Shapes will be coerced to a
            2D array of shape (n_trials, T). If lengths differ, truncate to the
            shortest length.
        n_bootstrap : int, optional (default: 2000)
            Number of bootstrap resamples.
        axis : int, optional (default: 0)
            Axis of trials in a provided 2D array. If axis==1, data is transposed.
        random_state : int or np.random.Generator, optional
            Seed or generator for reproducibility.

        Returns
        -------
        np.ndarray
            Bootstrap standard deviation of the mean per timestep (length T).
        """
        # import numpy as np

        if "trials_data" not in kwargs:
            raise ValueError("_bootstrapper requires 'trials_data' in kwargs.")

        trials_data = kwargs["trials_data"]
        n_boot = int(kwargs.get("n_bootstrap", 2000))
        axis = int(kwargs.get("axis", 0))
        rng = kwargs.get("random_state", None)
        rng = np.random.default_rng(rng)

        # Coerce to 2D array (n_trials, T)
        def _to_2d(data, axis=0):
            if isinstance(data, (list, tuple)):
                seq = [np.asarray(x, dtype=float) for x in data if x is not None]
                if not seq:
                    return np.zeros((0, 0), dtype=float)
                lengths = [len(x) for x in seq]
                if any(L != lengths[0] for L in lengths):
                    min_len = min(lengths)
                    seq = [x[:min_len] for x in seq]
                arr = np.vstack(seq)
                return arr if axis == 0 else arr.T

            arr = np.asarray(data, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            elif arr.ndim != 2:
                raise ValueError("trials_data must be list of 1D arrays or a 2D array.")
            return arr if axis == 0 else arr.T

        arr = _to_2d(trials_data, axis=axis)
        if arr.size == 0:
            return arr
        # Remove trials that are entirely NaN
        mask_keep = ~np.all(np.isnan(arr), axis=1)
        arr = arr[mask_keep]
        n_trials, T = arr.shape
        if n_trials < 2:
            # Not enough trials to estimate variability; return zeros
            return np.zeros(T, dtype=float)

        # Draw bootstrap indices and compute bootstrap means
        idx = rng.integers(0, n_trials, size=(n_boot, n_trials))
        boot_means = np.nanmean(arr[idx, :], axis=1)  # shape: (n_boot, T)
        boot_sd = np.nanstd(boot_means, axis=0, ddof=1)
        return boot_sd
        
  
