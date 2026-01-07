_STATS_READY = False

def ensure_statistics_registered():
    """Import metric module once so decorator registries are populated."""
    global _STATS_READY
    if _STATS_READY:
        return
    from ..metric import statistics_utils
    _STATS_READY = True

STATISTICS_REGISTRY = {}

def statistic(name=None):
    def deco(fn):
        if name is None:
            key = fn.__name__
        else:
            key = name
        # key = name or fn.__name__
        STATISTICS_REGISTRY[key] = fn
        return fn
    return deco
