import os
import pickle
from cellexp_util.registry.metric_registry import general_metric, synthetic_metric, GENERAL_METRICS, SYNTHETIC_METRICS

import numpy as np

@general_metric(name="tvNMSE", output="scalar")
def tvNMSE_metric(*, prediction, groundTruth, **_):
    s = groundTruth["s"]; y = prediction
    return ((y - s) ** 2).mean() / (s ** 2).mean()

@general_metric(name="tvMAE", output="scalar")
def tvMAE_metric(*, prediction, groundTruth, **_):
    s = groundTruth["s"]; y = prediction
    return np.abs(y - s).mean()

@general_metric(name="tvMAPE", output="scalar")
def tvMAPE_metric(*, prediction, groundTruth, **_):
    s = groundTruth["s"]; y = prediction
    m = s != 0
    return (np.abs((y[m] - s[m]) / s[m]).mean() * 100) if m.any() else np.nan

# stateful rolling NMSE – just use manager from kwargs
@general_metric(name="rollingNMSE", output="scalar")
def rollingNMSE_metric(*, prediction, groundTruth, manager, **_):
    y = groundTruth["s"]; yhat = prediction
    
    manager._cumulative_energy += y ** 2
    manager._nmse_n += (yhat - y) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        nmse_n = np.where(manager._cumulative_energy != 0,
                          manager._nmse_n / manager._cumulative_energy, 0.0)
    return float(nmse_n.mean())

# stateful rolling MAE – just use manager from kwargs
@general_metric(name="rollingMAE", output="scalar")
def rollingMAE_metric(*, manager, i, **_):
    try:
        return float(np.mean(manager.errors_MAE_single[:i + 1]))
    except Exception:
        return np.nan

# stateful rolling NMSE – just use manager from kwargs
@general_metric(name="rollingMAPE", output="scalar")
def rollingMAPE_metric(*, manager, i, **_):
    try:
        return float(np.mean(manager.errors_MAPE_single[:i + 1]))
    except Exception:
        return np.nan

def ROC_metric(groundTruth, estimated, wholeAdjacency, Nu):
    est = estimated[:Nu, :]
    wAd = wholeAdjacency[:,:Nu]
    gt = groundTruth[:,:Nu,:]

    gt = np.all(wAd[:,:,np.newaxis] == gt, axis = 0)

    detections = np.logical_and(gt, est)
    false_alarm = np.logical_and(np.logical_not(gt),est)

    pd = np.sum(detections)/np.sum(gt)
    pfa = np.sum(false_alarm)/np.sum(np.logical_not(gt))
    return pd, pfa

@synthetic_metric(name="pdpfa_lists", output="curve", single=False, to_delete=True)
def pdpfalists(*, thresholds, rfhorso, manager, groundTruthAdjacency, adjacencies, Nu, **_):
        
        manager.pdpfa_lists = []
        lists = ([],[])

        if thresholds is None:
            # import pdb
            # pdb.set_trace()
            return ([1.0],[1.0])
        for threshold in thresholds:
            b = rfhorso.extractUpperAdjacency(threshold)
            pd_val, pfa_val = ROC_metric(
                groundTruth=np.squeeze(groundTruthAdjacency[:, 1, :, :]),
                estimated=b,
                wholeAdjacency=np.squeeze(adjacencies[:, 1, :]),
                Nu=Nu,
            )
            lists[0].append(pd_val)
            lists[1].append(pfa_val)
        return lists

@synthetic_metric(name="pd", output="scalar")
def pd_metric(*, manager, **_):
        if not manager.pdpfa_lists:
            return np.nan
        pd_list = manager.pdpfa_lists[-1][0]
        if not pd_list:
            return np.nan
        mid = len(pd_list) // 2
        return pd_list[mid]
        

@synthetic_metric(name="pfa", output="scalar")
def pfa_metric(*, manager, **_):
        if not manager.pdpfa_lists:
            return np.nan
        pfa_list = manager.pdpfa_lists[-1][1]
        if not pfa_list:
            return np.nan
        mid = len(pfa_list) // 2
        return pfa_list[mid]

@synthetic_metric(name="pd_curve", output = "curve")
def pd_curve_metric(*, manager, **_):
    return manager.pdpfa_lists[0][0]

@synthetic_metric(name="pfa_curve", output="curve")
def pfa_curve_metric(*, manager, **_):
    return manager.pdpfa_lists[0][1]
    

class MetricManager:
    _registry = {}
    def __init__(self, N, T, savePath):
        self.populate_registry()
        self._T = T
        self._N = N
        self.reset_rolling()

        self._outputDir = savePath

        self._errors = {}
        for k, meta in self._registry.items():
            if meta["output"] == "scalar":
                self._errors[k] = np.zeros(T)
                if meta.get("single", True):
                    self._errors[k + "single"] = np.zeros(T)
            else:
                self._errors[k] = []
                if meta.get("single", True):
                    self._errors[k + "single"] = []

    def populate_registry(self):
        self._registry = GENERAL_METRICS

    def save_single(self, n):
        singleDict = {k : self._errors[k + "single"] for k, meta in self._registry.items() if meta.get("single", True)}
        errorPath = os.path.join(self._outputDir, f"errors_{n}.pkl")
        with open(errorPath, 'wb') as file:
            pickle.dump(singleDict, file)
            print(f"Errors are saved to {errorPath}")


    def reset_single(self):
       for k, meta in self._registry.items():
            if meta.get("single", True):
                if meta["output"] == "scalar":
                    self._errors[k + "single"] = np.zeros(self._T)
                else:
                    self._errors[k + "single"] = []


    def reset_rolling(self):
        """Reset rolling accumulators at the start of each repetition."""
        self._cumulative_energy = np.zeros(self._N)
        self._nmse_n = np.zeros(self._N)

    @property
    def cumulative_energy(self):
        return self._cumulative_energy
    
    @property
    def nmse_n(self):
        return self._nmse_n
    
    @cumulative_energy.setter
    def cumulative_energy(self,cum):
        self._cumulative_energy = cum

    @nmse_n.setter
    def nmse_n(self, nmse):
        self._nmse_n = nmse

    def step_calculation(self, i, prediction, groundTruth, verbose: bool = True, **kwargs):
        """Run all registered metrics for step i.
        Extra arguments are passed through via **kwargs so each metric can
        pull what it needs (e.g., thresholds, rfhorso, gt adjacencies, etc.).
        """
       
        for k, meta in self._registry.items():
    
            currError = meta["fn"](
                prediction=prediction,
                groundTruth=groundTruth,
                manager=self,
                i=i,
                **kwargs,
            )
           
            if meta["output"] == "scalar":
                if meta.get("single", True):
                    self._errors[k + "single"][i] = currError
                self._errors[k][i] += currError
                if verbose:
                    print(f"---- {k}: {currError}---")
            elif meta["output"] == "curve":
                if meta.get("single", True):
                    self._errors[k + "single"].append(currError)
                self._errors[k].append(currError)
            else:
                raise KeyError(f"Unknown output type for metric '{k}': {meta.get('output')}")
        
           

    def save_full(self, n):
        fullDict = dict()
        for k,meta in self._registry.items():
            if meta["to_delete"]:
                continue
            if meta["output"] == "scalar":
                fullDict[k] = self._errors[k] / n
            else:
                fullDict[k] = self._errors[k]
        
        errorPath = os.path.join(self._outputDir, "full_errors.pkl")
        with open(errorPath, 'wb') as file:
            pickle.dump(fullDict, file)
            print(f"Errors are saved to {file}")
        
    

    # @property
    # def errors_MAE_single(self):
    #     """
    #     Returns the errors dictionary containing all metrics.
    #     """
    #     return self._errors["tvMAEsingle"]
    # @property
    # def errors_MAPE_single(self):
    #     return self._errors["tvMAPEsingle"]
    


class SyntheticMetricManager(MetricManager):

    def __init__(self, N, T, savePath):
        super().__init__(N = N, T = T, savePath=savePath)


    def populate_registry(self):
        super().populate_registry()
        self._registry.update(SYNTHETIC_METRICS)

    @property
    def pdpfa_lists(self):
        return self._errors["pdpfa_lists"]
    
    @pdpfa_lists.setter
    def pdpfa_lists(self,pplist):
        self._errors["pdpfa_lists"] = pplist
    

   



