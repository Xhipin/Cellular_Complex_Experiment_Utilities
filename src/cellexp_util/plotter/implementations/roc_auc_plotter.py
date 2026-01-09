from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..plotter import PlotJobData, TableJobData


@dataclass
class ROCPlotConfig:
    output_dir: Optional[str] = None
    filename_template: str = "{experiment}_ROC"
    formats: Sequence[str] = ("pdf",)
    show: bool = False
    with_error: bool = True
    grid_points: int = 201
    tail_frac: float = 0.10
    error_method: str = "se"  # "se" or "bootstrap"
    n_bootstrap: int = 2000
    xlabel: str = "PFA"
    ylabel: str = "PD"
    title_template: Optional[str] = None


@dataclass
class AUCTableConfig:
    output_dir: Optional[str] = None
    filename_template: str = "{experiment}_AUC"
    formats: Sequence[str] = ("md", "tex")
    tail_frac: float = 0.10
    error_method: str = "se"  # "se" or "bootstrap"
    n_bootstrap: int = 2000
    number_format: str = "{:.3f}"
    missing: str = "-"


def render_roc_plot(job_data: PlotJobData, *, config: Optional[ROCPlotConfig] = None) -> List[str]:
    cfg = config or ROCPlotConfig()
    pfa_grid = np.linspace(0.0, 1.0, int(cfg.grid_points))
    y_map: Dict[str, np.ndarray] = {}
    err_map: Dict[str, np.ndarray] = {}

    for series in job_data.series:
        y_mean, y_err = _roc_series_from_artifacts(series.artifacts, pfa_grid, cfg)
        if y_mean is None:
            continue
        y_map[series.case.label] = y_mean
        if y_err is not None:
            err_map[series.case.label] = y_err

    if not y_map:
        return []

    return _plot_roc(
        x_values=pfa_grid,
        y_map=y_map,
        err_map=err_map if cfg.with_error else None,
        xlabel=cfg.xlabel,
        ylabel=cfg.ylabel,
        title=_format_title(cfg.title_template, job_data),
        output_dir=cfg.output_dir,
        filename=_format_filename(cfg.filename_template, job_data),
        formats=cfg.formats,
        show=cfg.show,
    )


def render_auc_table(job_data: TableJobData, *, config: Optional[AUCTableConfig] = None) -> Dict[str, str]:
    cfg = config or AUCTableConfig()
    headers: List[str] = []
    rows: List[List[str]] = []

    if job_data.cells:
        headers = [job_data.job.col_name] + [str(c) for c in job_data.job.col_values]
        for rv in job_data.job.row_values:
            row = [str(rv)]
            for cv in job_data.job.col_values:
                cell = job_data.cells.get((rv, cv))
                row.append(_format_auc_cell(cell, cfg))
            rows.append(row)
    elif job_data.algs:
        algs = list(job_data.algs.keys())
        if job_data.job.row_var == "alg":
            headers = [job_data.job.row_name, "AUC"]
            for alg in algs:
                row = [alg, _format_auc_from_artifact(job_data.algs.get(alg), cfg)]
                rows.append(row)
        else:
            headers = [job_data.job.row_name] + algs
            row = ["AUC"]
            for alg in algs:
                row.append(_format_auc_from_artifact(job_data.algs.get(alg), cfg))
            rows.append(row)
    else:
        return {}

    md = _format_markdown(headers, rows)
    tex = _format_latex(headers, rows)

    if cfg.output_dir:
        base = _format_filename(cfg.filename_template, job_data)
        for fmt in cfg.formats:
            fmt_l = fmt.lower()
            if fmt_l == "md":
                _write_text(_join_out(cfg.output_dir, base, "md"), md)
            elif fmt_l == "tex":
                _write_text(_join_out(cfg.output_dir, base, "tex"), tex)

    return {"md": md, "tex": tex}


def _roc_series_from_artifacts(artifacts, pfa_grid: np.ndarray, cfg: ROCPlotConfig):
    trial_pd_on_grid: List[np.ndarray] = []
    for errs in artifacts.trial_errors:
        pd_vec, pfa_vec = _trial_pd_pfa(errs, cfg.tail_frac)
        if pd_vec is None or pfa_vec is None:
            continue
        pd_interp = _interp_pd_on_grid(pd_vec, pfa_vec, pfa_grid)
        if pd_interp is not None:
            trial_pd_on_grid.append(pd_interp)

    if trial_pd_on_grid:
        arr = np.vstack(trial_pd_on_grid)
        y_mean = np.nanmean(arr, axis=0)
        y_err = _error_from_trials(arr, cfg)
        return y_mean, y_err

    if artifacts.full_errors:
        pd_vec, pfa_vec = _trial_pd_pfa(artifacts.full_errors, cfg.tail_frac, single=False)
        if pd_vec is not None and pfa_vec is not None:
            y_mean = _interp_pd_on_grid(pd_vec, pfa_vec, pfa_grid)
            return y_mean, None

    return None, None


def _trial_pd_pfa(errs: Dict, tail_frac: float, single: bool = True):
    if single:
        pd_list = errs.get("pd_curvesingle") or errs.get("pd_curve")
        pfa_list = errs.get("pfa_curvesingle") or errs.get("pfa_curve")
    else:
        pd_list = errs.get("pd_curve") or errs.get("pd_curvesingle")
        pfa_list = errs.get("pfa_curve") or errs.get("pfa_curvesingle")
    pd_vec = _tail_average_curve_list(pd_list, tail_frac)
    pfa_vec = _tail_average_curve_list(pfa_list, tail_frac)
    if pd_vec is None or pfa_vec is None:
        return None, None
    L = min(len(pd_vec), len(pfa_vec))
    if L == 0:
        return None, None
    pd_vec = np.asarray(pd_vec[:L], dtype=float)
    pfa_vec = np.asarray(pfa_vec[:L], dtype=float)
    return pd_vec, pfa_vec


def _tail_average_curve_list(seq_of_curves, frac: float) -> Optional[np.ndarray]:
    try:
        if not isinstance(seq_of_curves, (list, tuple)) or not seq_of_curves:
            return None
        k = max(1, int(math.ceil(frac * len(seq_of_curves))))
        tail = seq_of_curves[-k:]
        arrs = [np.asarray(x, dtype=float) for x in tail if x is not None]
        if not arrs:
            return None
        L = min(a.size for a in arrs)
        if L == 0:
            return None
        arrs = [a[:L] for a in arrs]
        return np.vstack(arrs).mean(axis=0)
    except Exception:
        return None


def _interp_pd_on_grid(pd_vec: np.ndarray, pfa_vec: np.ndarray, grid: np.ndarray) -> Optional[np.ndarray]:
    try:
        order = np.argsort(pfa_vec)
        pfa_sorted = pfa_vec[order]
        pd_sorted = pd_vec[order]
        return np.interp(grid, pfa_sorted, pd_sorted, left=pd_sorted[0], right=pd_sorted[-1])
    except Exception:
        return None


def _error_from_trials(arr: np.ndarray, cfg: ROCPlotConfig) -> Optional[np.ndarray]:
    if arr.shape[0] < 2:
        return None
    if cfg.error_method == "bootstrap":
        return _bootstrap_se(arr, cfg.n_bootstrap)
    return _standard_error(arr)


def _standard_error(arr: np.ndarray) -> np.ndarray:
    n_eff = np.sum(~np.isnan(arr), axis=0)
    std = np.nanstd(arr, axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return std / np.sqrt(n_eff)


def _bootstrap_se(arr: np.ndarray, n_bootstrap: int) -> np.ndarray:
    n_trials, t = arr.shape
    if n_trials < 2:
        return np.zeros(t, dtype=float)
    rng = np.random.default_rng()
    idx = rng.integers(0, n_trials, size=(n_bootstrap, n_trials))
    boot_means = np.nanmean(arr[idx, :], axis=1)
    return np.nanstd(boot_means, axis=0, ddof=1)


def _format_auc_cell(cell, cfg: AUCTableConfig) -> str:
    if cell is None:
        return cfg.missing
    values: List[float] = []
    for art in cell.artifacts:
        values.extend(_auc_values_from_artifact(art, cfg))
    return _format_value_list(values, cfg)


def _format_auc_from_artifact(art, cfg: AUCTableConfig) -> str:
    if art is None:
        return cfg.missing
    values = _auc_values_from_artifact(art, cfg)
    return _format_value_list(values, cfg)


def _auc_values_from_artifact(art, cfg: AUCTableConfig) -> List[float]:
    values: List[float] = []
    if art.trial_errors:
        for errs in art.trial_errors:
            auc = _trial_auc(errs, cfg.tail_frac)
            if auc is not None and np.isfinite(auc):
                values.append(float(auc))
    elif art.full_errors:
        auc = _trial_auc(art.full_errors, cfg.tail_frac, single=False)
        if auc is not None and np.isfinite(auc):
            values.append(float(auc))
    return values


def _trial_auc(errs: Dict, tail_frac: float, single: bool = True) -> Optional[float]:
    pd_vec, pfa_vec = _trial_pd_pfa(errs, tail_frac, single=single)
    if pd_vec is None or pfa_vec is None:
        return None
    order = np.argsort(pfa_vec)
    try:
        return float(np.trapz(pd_vec[order], pfa_vec[order]))
    except Exception:
        return float(np.trapz(pd_vec, pfa_vec))


def _format_value_list(values: List[float], cfg: AUCTableConfig) -> str:
    if not values:
        return cfg.missing
    mean = float(np.nanmean(values))
    err = _error_from_scalar_trials(values, cfg)
    if err is None:
        return cfg.number_format.format(mean)
    return f"{cfg.number_format.format(mean)} +/- {cfg.number_format.format(err)}"


def _error_from_scalar_trials(values: List[float], cfg: AUCTableConfig) -> Optional[float]:
    if len(values) < 2:
        return None
    arr = np.asarray(values, dtype=float)
    if cfg.error_method == "bootstrap":
        rng = np.random.default_rng()
        n = len(arr)
        idx = rng.integers(0, n, size=(cfg.n_bootstrap, n))
        boot_means = np.nanmean(arr[idx], axis=1)
        return float(np.nanstd(boot_means, ddof=1))
    std = float(np.nanstd(arr, ddof=1))
    return std / math.sqrt(len(arr))


def _plot_roc(
    *,
    x_values: np.ndarray,
    y_map: Dict[str, np.ndarray],
    err_map: Optional[Dict[str, np.ndarray]],
    xlabel: str,
    ylabel: str,
    title: Optional[str],
    output_dir: Optional[str],
    filename: str,
    formats: Sequence[str],
    show: bool,
) -> List[str]:
    import matplotlib.pyplot as plt
    import itertools

    saved: List[str] = []
    linestyles = itertools.cycle(["-", "--", "-.", ":"])

    fig, ax = plt.subplots(figsize=(6, 6))
    for label, y in y_map.items():
        ls = next(linestyles)
        ax.plot(x_values, y, label=label, linewidth=2, linestyle=ls)
        if err_map and label in err_map:
            err = err_map[label]
            if err is not None and len(err) == len(y):
                y_lo = y - err
                y_hi = y + err
                ax.fill_between(x_values, y_lo, y_hi, alpha=0.2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(ncol=1)
    fig.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for fmt in formats:
            path = _join_out(output_dir, filename, fmt)
            fig.savefig(path, bbox_inches="tight")
            saved.append(path)
    if show and not output_dir:
        plt.show()
    plt.close(fig)
    return saved


def _format_title(template: Optional[str], job_data: PlotJobData) -> Optional[str]:
    if not template:
        return None
    return template.format(
        experiment=job_data.job.experiment_name,
        iterator=job_data.job.iterator_var,
        iter_value=job_data.job.iterator_value,
    )


def _format_filename(template: str, job_data) -> str:
    ctx = {
        "experiment": job_data.job.experiment_name,
        "iterator": getattr(job_data.job, "iterator_var", None),
        "iter_value": getattr(job_data.job, "iterator_value", None),
    }
    try:
        return template.format(**ctx)
    except Exception:
        return f"{job_data.job.experiment_name}_ROC"


def _format_markdown(headers: List[str], rows: List[List[str]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([line, sep] + body)


def _format_latex(headers: List[str], rows: List[List[str]]) -> str:
    cols = "l" + "c" * (len(headers) - 1)
    lines = [r"\begin{tabular}{" + cols + r"}", r"\hline"]
    lines.append(" & ".join(headers) + r" \\")
    lines.append(r"\hline")
    for r in rows:
        lines.append(" & ".join(r) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def _join_out(out_dir: str, base: str, ext: str) -> str:
    return os.path.join(out_dir, f"{base}.{ext}")


def _write_text(path: str, content: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        f.write(content + "\n")
