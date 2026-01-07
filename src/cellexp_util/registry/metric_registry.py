import numpy as np
_METRICS_READY = False

def ensure_metrics_registered():
    """Import metric module once so decorator registries are populated."""
    global _METRICS_READY
    if _METRICS_READY:
        return
    from ..metric import metric_utils  # triggers decorator registration
    _METRICS_READY = True

# ---- global registries + decorators ---------------------------------

GENERAL_METRICS = {}     # name -> {"fn": callable, "output": "scalar" | "curve"}


SYNTHETIC_METRICS = {}   # same shape

def synthetic_metric(name=None, output="scalar", to_delete = False, single = True):
    def deco(fn):
        key = name or fn.__name__.lstrip("_")
        SYNTHETIC_METRICS[key] = {"fn": fn, "output": output, "to_delete": to_delete, "single": single}
        return fn
    return deco


def general_metric(name=None, output="scalar", to_delete = False, single = True):
        """
        Register a general metric.
        output: "scalar" => per-timestep scalar (array storage)
                "curve"  => per-timestep list/curve (list storage)
        """
        def deco(fn):
            key = name or fn.__name__.lstrip("_")
            GENERAL_METRICS[key] = {"fn": fn, "output": output, "to_delete": to_delete, "single": single}
            return fn
        return deco


def _metric_info(name: str):
    if name is None:
        return None
    return GENERAL_METRICS.get(name) or SYNTHETIC_METRICS.get(name)


def _is_curve_metric(name: str) -> bool:
    info = _metric_info(name)
    return bool(info and info.get("output") == "curve")


def _resolve_metric(errs_dict, canonical_name):
    """Resolve a canonical metric name to a 1-D numeric array from an errors dict.
    Uses the registry to skip curve-type metrics and tries both the exact key and
    the "single" variant (e.g., tvNMSE and tvNMSEsingle).
    """
    if canonical_name is None:
        return None
    name = str(canonical_name)
    if _is_curve_metric(name):
        return None
    # try exact
    if name in errs_dict:
        arr = np.asarray(errs_dict[name])
        if getattr(arr, "ndim", 0) == 1 and np.issubdtype(arr.dtype, np.number):
            return arr
    # try the "single" suffix
    single_key = f"{name}single"
    if single_key in errs_dict:
        arr = np.asarray(errs_dict[single_key])
        if getattr(arr, "ndim", 0) == 1 and np.issubdtype(arr.dtype, np.number):
            return arr
    return None
