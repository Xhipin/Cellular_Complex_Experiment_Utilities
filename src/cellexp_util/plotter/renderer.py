from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..registry.metric_registry import (
    ensure_metrics_registered,
    _is_curve_metric,
    _resolve_metric,
)
from .plotter import PlotJobData, TableJobData


@dataclass
class PlotRenderConfig:
    output_dir: Optional[str] = None
    filename_template: str = "{experiment}_{metric}"
    formats: Sequence[str] = ("pdf",)
    show: bool = False
    with_error: bool = True
    error_method: str = "se"  # "se" or "bootstrap"
    n_bootstrap: int = 2000
    xlabel: str = "t"
    ylabel_template: str = "{metric}"
    title_template: Optional[str] = None


@dataclass
class TableRenderConfig:
    output_dir: Optional[str] = None
    filename_template: str = "{experiment}_{metric}"
    formats: Sequence[str] = ("md", "tex")
    tail_frac: float = 0.10
    error_method: str = "se"  # "se" or "bootstrap"
    n_bootstrap: int = 2000
    number_format: str = "{:.3f}"
    missing: str = "-"


def render_plot_job(
    job_data: PlotJobData,
    *,
    metrics: Optional[Sequence[str]] = None,
    config: Optional[PlotRenderConfig] = None,
) -> List[str]:
    ensure_metrics_registered()
    cfg = config or PlotRenderConfig()
    metrics_to_plot = list(metrics) if metrics is not None else _discover_metrics(job_data.series)
    saved: List[str] = []

    for metric in metrics_to_plot:
        x_values, y_map, err_map = _build_plot_series(job_data, metric, cfg)
        if not y_map:
            continue
        fig_paths = _plot_series(
            x_values=x_values,
            y_map=y_map,
            err_map=err_map if cfg.with_error else None,
            xlabel=cfg.xlabel,
            ylabel=cfg.ylabel_template.format(metric=metric),
            title=_format_title(cfg.title_template, job_data, metric),
            output_dir=cfg.output_dir,
            filename=_format_filename(cfg.filename_template, job_data, metric),
            formats=cfg.formats,
            show=cfg.show,
        )
        saved.extend(fig_paths)
    return saved


def render_table_job(
    job_data: TableJobData,
    *,
    metrics: Optional[Sequence[str]] = None,
    config: Optional[TableRenderConfig] = None,
) -> Dict[str, Dict[str, str]]:
    ensure_metrics_registered()
    cfg = config or TableRenderConfig()
    metrics_to_use = list(metrics) if metrics is not None else _discover_table_metrics(job_data)
    outputs: Dict[str, Dict[str, str]] = {}

    if job_data.algs and job_data.job.row_var == "metric":
        table = _build_comparison_table(job_data, metrics_to_use, cfg)
        if table is not None:
            headers, rows = table
            md = _format_markdown(headers, rows)
            tex = _format_latex(headers, rows)
            outputs["metrics"] = {"md": md, "tex": tex}

            if cfg.output_dir:
                base = _format_filename(cfg.filename_template, job_data, "metrics")
                for fmt in cfg.formats:
                    fmt_l = fmt.lower()
                    if fmt_l == "md":
                        _write_text(_join_out(cfg.output_dir, base, "md"), md)
                    elif fmt_l == "tex":
                        _write_text(_join_out(cfg.output_dir, base, "tex"), tex)
        return outputs

    for metric in metrics_to_use:
        table = _build_table(job_data, metric, cfg)
        if table is None:
            continue
        headers, rows = table
        md = _format_markdown(headers, rows)
        tex = _format_latex(headers, rows)
        outputs[metric] = {"md": md, "tex": tex}

        if cfg.output_dir:
            base = _format_filename(cfg.filename_template, job_data, metric)
            for fmt in cfg.formats:
                fmt_l = fmt.lower()
                if fmt_l == "md":
                    _write_text(_join_out(cfg.output_dir, base, "md"), md)
                elif fmt_l == "tex":
                    _write_text(_join_out(cfg.output_dir, base, "tex"), tex)
    return outputs


def _discover_metrics(series: Iterable) -> List[str]:
    seen = set()
    metrics: List[str] = []
    for s in series:
        errs = s.artifacts.full_errors or {}
        for key in errs.keys():
            if not isinstance(key, str):
                continue
            canon = key[:-6] if key.endswith("single") else key
            if _is_curve_metric(canon):
                continue
            if canon not in seen:
                metrics.append(canon)
                seen.add(canon)
    return metrics


def _discover_table_metrics(job_data: TableJobData) -> List[str]:
    metrics: List[str] = []
    seen = set()
    if job_data.cells:
        for cell in job_data.cells.values():
            for art in cell.artifacts:
                errs = art.full_errors or {}
                for key in errs.keys():
                    if not isinstance(key, str):
                        continue
                    canon = key[:-6] if key.endswith("single") else key
                    if _is_curve_metric(canon):
                        continue
                    if canon not in seen:
                        metrics.append(canon)
                        seen.add(canon)
    elif job_data.algs:
        for art in job_data.algs.values():
            errs = art.full_errors or {}
            for key in errs.keys():
                if not isinstance(key, str):
                    continue
                canon = key[:-6] if key.endswith("single") else key
                if _is_curve_metric(canon):
                    continue
                if canon not in seen:
                    metrics.append(canon)
                    seen.add(canon)
    return metrics


def _build_plot_series(job_data: PlotJobData, metric: str, cfg: PlotRenderConfig):
    y_map: Dict[str, np.ndarray] = {}
    err_map: Dict[str, np.ndarray] = {}
    x_values: Optional[np.ndarray] = None

    for series in job_data.series:
        y = _resolve_metric_from_artifacts(series.artifacts, metric)
        if y is None:
            continue
        y = np.asarray(y, dtype=float)
        if x_values is None:
            x_values = np.arange(y.size)
        y_map[series.case.label] = y

        err = _error_from_trials(series.artifacts, metric, cfg)
        if err is not None:
            err_map[series.case.label] = err

    if x_values is None:
        x_values = np.array([])
    return x_values, y_map, err_map


def _resolve_metric_from_artifacts(artifacts, metric: str) -> Optional[np.ndarray]:
    if artifacts.full_errors:
        y = _resolve_metric(artifacts.full_errors, metric)
        if y is not None:
            return y
    if artifacts.trial_errors:
        arr = _stack_trials([_resolve_metric(e, metric) for e in artifacts.trial_errors])
        if arr is not None and arr.size:
            return np.nanmean(arr, axis=0)
    return None


def _error_from_trials(artifacts, metric: str, cfg: PlotRenderConfig) -> Optional[np.ndarray]:
    if not artifacts.trial_errors:
        return None
    arr = _stack_trials([_resolve_metric(e, metric) for e in artifacts.trial_errors])
    if arr is None or arr.shape[0] < 2:
        return None
    if cfg.error_method == "bootstrap":
        return _bootstrap_se(arr, n_bootstrap=cfg.n_bootstrap)
    return _standard_error(arr)


def _stack_trials(trials: Iterable[Optional[np.ndarray]]) -> Optional[np.ndarray]:
    seq = [np.asarray(x, dtype=float) for x in trials if x is not None]
    if not seq:
        return None
    min_len = min(len(x) for x in seq)
    if min_len == 0:
        return None
    seq = [x[:min_len] for x in seq]
    return np.vstack(seq)


def _standard_error(arr: np.ndarray) -> np.ndarray:
    n_eff = np.sum(~np.isnan(arr), axis=0)
    std = np.nanstd(arr, axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        se = std / np.sqrt(n_eff)
    return se


def _bootstrap_se(arr: np.ndarray, n_bootstrap: int = 2000) -> np.ndarray:
    n_trials, t = arr.shape
    if n_trials < 2:
        return np.zeros(t, dtype=float)
    rng = np.random.default_rng()
    idx = rng.integers(0, n_trials, size=(n_bootstrap, n_trials))
    boot_means = np.nanmean(arr[idx, :], axis=1)
    return np.nanstd(boot_means, axis=0, ddof=1)


def _plot_series(
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

    fig, ax = plt.subplots(figsize=(6, 4))
    for label, y in y_map.items():
        ls = next(linestyles)
        ax.plot(x_values, y, label=label, linewidth=2, linestyle=ls)
        if err_map and label in err_map:
            err = err_map[label]
            if err is not None and len(err) == len(y):
                y_lo = y - err
                y_hi = y + err
                ax.fill_between(x_values, y_lo, y_hi, alpha=0.2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(ncol=2)
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


def _format_title(template: Optional[str], job_data: PlotJobData, metric: str) -> Optional[str]:
    if not template:
        return None
    return template.format(
        experiment=job_data.job.experiment_name,
        metric=metric,
        iterator=job_data.job.iterator_var,
        iter_value=job_data.job.iterator_value,
    )


def _build_table(
    job_data: TableJobData, metric: str, cfg: TableRenderConfig
) -> Optional[Tuple[List[str], List[List[str]]]]:
    if job_data.cells:
        headers = [job_data.job.col_name] + [str(c) for c in job_data.job.col_values]
        rows: List[List[str]] = []
        for rv in job_data.job.row_values:
            row = [str(rv)]
            for cv in job_data.job.col_values:
                cell = job_data.cells.get((rv, cv))
                row.append(_format_cell(cell, metric, cfg))
            rows.append(row)
        return headers, rows

    if job_data.algs:
        algs = list(job_data.algs.keys())
        headers = [job_data.job.row_name] + algs
        rows: List[List[str]] = []
        for m in [metric]:
            row = [m]
            for alg in algs:
                art = job_data.algs.get(alg)
                cell = None
                if art is not None:
                    cell = TableCellData(cases=[], artifacts=[art])
                row.append(_format_cell(cell, m, cfg))
            rows.append(row)
        return headers, rows

    return None


def _build_comparison_table(
    job_data: TableJobData, metrics: List[str], cfg: TableRenderConfig
) -> Optional[Tuple[List[str], List[List[str]]]]:
    if not job_data.algs:
        return None
    algs = list(job_data.algs.keys())
    headers = [job_data.job.row_name] + algs
    rows: List[List[str]] = []
    for m in metrics:
        row = [m]
        for alg in algs:
            art = job_data.algs.get(alg)
            cell = None
            if art is not None:
                cell = TableCellData(cases=[], artifacts=[art])
            row.append(_format_cell(cell, m, cfg))
        rows.append(row)
    return headers, rows


def _format_cell(cell: Optional[TableCellData], metric: str, cfg: TableRenderConfig) -> str:
    if cell is None:
        return cfg.missing
    values: List[float] = []
    if cell.artifacts:
        for art in cell.artifacts:
            if art.trial_errors:
                trials = [_resolve_metric(e, metric) for e in art.trial_errors]
                arr = _stack_trials(trials)
                if arr is not None and arr.size:
                    for row in arr:
                        values.append(_tail_mean(row, cfg.tail_frac))
            elif art.full_errors:
                y = _resolve_metric(art.full_errors, metric)
                if y is not None:
                    values.append(_tail_mean(np.asarray(y, dtype=float), cfg.tail_frac))
    if not values:
        return cfg.missing
    mean = float(np.nanmean(values))
    err = _error_from_scalar_trials(values, cfg)
    if err is None:
        return cfg.number_format.format(mean)
    return f"{cfg.number_format.format(mean)} +/- {cfg.number_format.format(err)}"


def _tail_mean(arr: np.ndarray, frac: float) -> float:
    if arr.size == 0:
        return float("nan")
    k = max(1, int(math.ceil(frac * arr.size)))
    return float(np.nanmean(arr[-k:]))


def _error_from_scalar_trials(values: List[float], cfg: TableRenderConfig) -> Optional[float]:
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


def _format_filename(template: str, job_data, metric: str) -> str:
    ctx = {
        "experiment": job_data.job.experiment_name,
        "metric": metric,
        "iterator": getattr(job_data.job, "iterator_var", None),
        "iter_value": getattr(job_data.job, "iterator_value", None),
    }
    try:
        return template.format(**ctx)
    except Exception:
        return f"{job_data.job.experiment_name}_{metric}"
