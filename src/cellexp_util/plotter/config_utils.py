from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml


# ---------------------------
# Manifest data structures
# ---------------------------


@dataclass(frozen=True)
class SweepSpec:
    vars: List[str]
    values: List[List[Any]]
    sweep_type: str = "cartesian"

    def expand(self) -> List[Dict[str, Any]]:
        if self.sweep_type == "cartesian":
            return _expand_cartesian(self.vars, self.values)
        if self.sweep_type == "linear":
            return _expand_linear(self.vars, self.values)
        raise ValueError(f"Unknown sweep_type: {self.sweep_type}")


@dataclass(frozen=True)
class PlotSweepRule:
    iterator: Optional[str]
    legend: str
    legend_name: str
    iterator_values: Optional[List[Any]]
    legend_values: Optional[List[Any]]
    extra_vars: List[str] = field(default_factory=list)
    extra_values: List[Any] = field(default_factory=list)


@dataclass(frozen=True)
class TableSweepRule:
    column_var: str
    row_var: str
    column_name: str
    row_name: str
    extra_vars: List[str] = field(default_factory=list)
    extra_values: List[Any] = field(default_factory=list)


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    path: str
    label: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResolvedCase:
    label: str
    path: str
    meta: Dict[str, Any]
    alg: Optional[str] = None


@dataclass
class PlotJob:
    experiment_name: str
    iterator_var: Optional[str]
    iterator_value: Any
    legend_var: str
    legend_name: str
    extra_vars: List[str]
    extra_values: List[Any]
    curves: List[ResolvedCase]


@dataclass
class TableJob:
    experiment_name: str
    row_var: str
    col_var: str
    row_name: str
    col_name: str
    row_values: List[Any]
    col_values: List[Any]
    row_index: Dict[Any, int]
    col_index: Dict[Any, int]
    extra_vars: List[str]
    extra_values: List[Any]
    cells: Dict[Tuple[Any, Any], List[ResolvedCase]]
    algs: List[str] = field(default_factory=list)
    alg_paths: Dict[str, str] = field(default_factory=dict)


@dataclass
class AblationManifest:
    algorithm: str
    base_path: str
    path_template: str
    label_template: Optional[str]
    sweeps: Dict[str, SweepSpec]
    plot_sweeps: Dict[str, List[PlotSweepRule]]
    table_sweeps: Dict[str, List[TableSweepRule]]

    _configs_cache: Optional[List[Dict[str, Any]]] = field(default=None, init=False, repr=False)

    def expand_sweeps(self) -> List[Dict[str, Any]]:
        if self._configs_cache is not None:
            return list(self._configs_cache)
        flattened: List[Dict[str, Any]] = []
        for exp_name, sweep in self.sweeps.items():
            for cfg in sweep.expand():
                flattened.append({"experiment_name": exp_name, **cfg})
        self._configs_cache = flattened
        return list(flattened)

    def plot_jobs(self) -> List[PlotJob]:
        configs = self.expand_sweeps()
        jobs: List[PlotJob] = []
        for exp_name, rules in self.plot_sweeps.items():
            for rule in rules:
                iterator_var = rule.iterator
                legend_var = rule.legend
                legend_name = rule.legend_name

                iter_values = rule.iterator_values
                if iter_values is None:
                    iter_values = [None]
                legend_values = rule.legend_values or []

                for iter_val in iter_values:
                    curves: List[ResolvedCase] = []
                    for leg_val in legend_values:
                        match = _find_config(
                            configs=configs,
                            exp_name=exp_name,
                            iterator_var=iterator_var,
                            iterator_val=iter_val,
                            legend_var=legend_var,
                            legend_val=leg_val,
                            extra_vars=rule.extra_vars,
                            extra_values=rule.extra_values,
                        )
                        if match is None:
                            continue
                        path = _resolve_path(self.base_path, self.path_template, match)
                        label = _format_label(legend_name, leg_val, match, self.label_template)
                        curves.append(
                            ResolvedCase(label=label, path=path, meta=match, alg=self.algorithm)
                        )
                    if curves:
                        jobs.append(
                            PlotJob(
                                experiment_name=exp_name,
                                iterator_var=iterator_var,
                                iterator_value=iter_val,
                                legend_var=legend_var,
                                legend_name=legend_name,
                                extra_vars=list(rule.extra_vars),
                                extra_values=list(rule.extra_values),
                                curves=curves,
                            )
                        )
        return jobs

    def table_jobs(self) -> List[TableJob]:
        configs = self.expand_sweeps()
        jobs: List[TableJob] = []
        for exp_name, specs in self.table_sweeps.items():
            for spec in specs:
                row_var = spec.row_var
                col_var = spec.column_var
                row_name = str(spec.row_name)
                col_name = str(spec.column_name)
                extra_vars = spec.extra_vars
                extra_values = spec.extra_values

                filtered = _filter_configs(
                    configs=configs,
                    exp_name=exp_name,
                    extra_vars=extra_vars,
                    extra_values=extra_values,
                    required_vars=[row_var, col_var],
                )
                if not filtered:
                    continue

                row_values, col_values = _unique_values(filtered, row_var, col_var)
                row_index = {v: i for i, v in enumerate(row_values)}
                col_index = {v: i for i, v in enumerate(col_values)}
                cells = {(rv, cv): [] for rv in row_values for cv in col_values}

                for cfg in filtered:
                    rv, cv = cfg[row_var], cfg[col_var]
                    path = _resolve_path(self.base_path, self.path_template, cfg)
                    label = _format_label(None, None, cfg, self.label_template)
                    cells[(rv, cv)].append(
                        ResolvedCase(label=label, path=path, meta=cfg, alg=self.algorithm)
                    )

                jobs.append(
                    TableJob(
                        experiment_name=exp_name,
                        row_var=row_var,
                        col_var=col_var,
                        row_name=row_name,
                        col_name=col_name,
                        row_values=row_values,
                        col_values=col_values,
                        row_index=row_index,
                        col_index=col_index,
                        extra_vars=list(extra_vars),
                        extra_values=list(extra_values),
                        cells=cells,
                        algs=[self.algorithm],
                        alg_paths={self.algorithm: self.base_path},
                    )
                )
        return jobs


@dataclass
class ComparisonManifest:
    algorithms: List[AlgorithmSpec]
    plot_sweeps: Dict[str, List[PlotSweepRule]]
    table_sweeps: Dict[str, List[TableSweepRule]]

    def plot_jobs(self) -> List[PlotJob]:
        jobs: List[PlotJob] = []
        alg_by_name = {a.name: a for a in self.algorithms}
        for exp_name, rules in self.plot_sweeps.items():
            for rule in rules:
                if rule.iterator is not None:
                    raise ValueError("Comparison plots only support iterator: null.")
                legend_values = rule.legend_values or list(alg_by_name.keys())
                curves: List[ResolvedCase] = []
                for leg_val in legend_values:
                    alg = alg_by_name.get(leg_val)
                    if alg is None:
                        continue
                    label_val = alg.label or alg.name
                    label = _format_label(rule.legend_name, label_val, alg.meta, None)
                    curves.append(
                        ResolvedCase(label=label, path=alg.path, meta=alg.meta, alg=alg.name)
                    )
                if curves:
                    jobs.append(
                        PlotJob(
                            experiment_name=exp_name,
                            iterator_var=None,
                            iterator_value=None,
                            legend_var=rule.legend,
                            legend_name=rule.legend_name,
                            extra_vars=list(rule.extra_vars),
                            extra_values=list(rule.extra_values),
                            curves=curves,
                        )
                    )
        return jobs

    def table_jobs(self) -> List[TableJob]:
        jobs: List[TableJob] = []
        algs = [a.name for a in self.algorithms]
        alg_paths = {a.name: a.path for a in self.algorithms}
        for exp_name, specs in self.table_sweeps.items():
            for spec in specs:
                jobs.append(
                    TableJob(
                        experiment_name=exp_name,
                        row_var=spec.row_var,
                        col_var=spec.column_var,
                        row_name=str(spec.row_name),
                        col_name=str(spec.column_name),
                        row_values=[],
                        col_values=list(algs),
                        row_index={},
                        col_index={a: i for i, a in enumerate(algs)},
                        extra_vars=list(spec.extra_vars),
                        extra_values=list(spec.extra_values),
                        cells={},
                        algs=list(algs),
                        alg_paths=dict(alg_paths),
                    )
                )
        return jobs


# ---------------------------
# YAML parsing
# ---------------------------


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping/dict.")
    return data


def load_manifest(path: str) -> AblationManifest | ComparisonManifest:
    data = load_yaml(path)
    if "sweeps" in data or "algorithm" in data:
        return AblationManifest(
            algorithm=data["algorithm"],
            base_path=data.get("base_path", ""),
            path_template=data.get("path_template", ""),
            label_template=data.get("label_template"),
            sweeps=_parse_sweeps(data.get("sweeps", {})),
            plot_sweeps=_parse_plot_sweeps(data.get("plot_sweeps", {})),
            table_sweeps=_parse_table_sweeps(data.get("table_sweeps", {})),
        )
    if "algorithms" in data:
        return ComparisonManifest(
            algorithms=_parse_algorithms(data.get("algorithms", [])),
            plot_sweeps=_parse_plot_sweeps(data.get("plot_sweeps", {})),
            table_sweeps=_parse_table_sweeps(data.get("table_sweeps", {})),
        )
    raise ValueError("YAML must define either 'sweeps'/'algorithm' or 'algorithms'.")


# ---------------------------
# Helpers
# ---------------------------


def _expand_cartesian(vars: List[str], values: List[List[Any]]) -> List[Dict[str, Any]]:
    if len(vars) != len(values):
        raise ValueError("vars and values must have the same length.")
    configs = []
    for combo in itertools.product(*values):
        configs.append({vars[i]: combo[i] for i in range(len(vars))})
    return configs


def _expand_linear(vars: List[str], values: List[List[Any]]) -> List[Dict[str, Any]]:
    if len(vars) != len(values):
        raise ValueError("vars and values must have the same length.")
    lengths = [len(v) for v in values]
    if len(set(lengths)) != 1:
        raise ValueError("All value lists must have the same length for linear sweeps.")
    configs = []
    for i in range(lengths[0]):
        configs.append({vars[j]: values[j][i] for j in range(len(vars))})
    return configs


def _parse_sweeps(sweeps: dict) -> Dict[str, SweepSpec]:
    out: Dict[str, SweepSpec] = {}
    for exp_name, meta in sweeps.items():
        out[exp_name] = SweepSpec(
            vars=list(meta.get("vars", [])),
            values=list(meta.get("values", [])),
            sweep_type=meta.get("sweep_type", "cartesian"),
        )
    return out


def _parse_plot_sweeps(plot_sweeps: dict) -> Dict[str, List[PlotSweepRule]]:
    out: Dict[str, List[PlotSweepRule]] = {}
    for exp_name, rules in plot_sweeps.items():
        parsed_rules: List[PlotSweepRule] = []
        for rule in rules:
            if isinstance(rule, (list, tuple)) and len(rule) == 2:
                iterator_spec, values_spec = rule
            else:
                iterator_spec, values_spec = rule, rule
            parsed_rules.append(
                PlotSweepRule(
                    iterator=iterator_spec.get("iterator"),
                    legend=iterator_spec.get("legend"),
                    legend_name=iterator_spec.get("legend_name", "{}"),
                    iterator_values=values_spec.get("iterator_values"),
                    legend_values=values_spec.get("legend_values"),
                    extra_vars=iterator_spec.get("extra_vars", []),
                    extra_values=values_spec.get("extra_values", []),
                )
            )
        out[exp_name] = parsed_rules
    return out


def _parse_table_sweeps(table_sweeps: dict) -> Dict[str, List[TableSweepRule]]:
    out: Dict[str, List[TableSweepRule]] = {}
    for exp_name, specs in table_sweeps.items():
        parsed_specs: List[TableSweepRule] = []
        for spec in specs:
            parsed_specs.append(
                TableSweepRule(
                    column_var=spec.get("column_var"),
                    row_var=spec.get("row_var"),
                    column_name=spec.get("column_name", "{}"),
                    row_name=spec.get("row_name", "{}"),
                    extra_vars=spec.get("extra_vars", []),
                    extra_values=spec.get("extra_values", []),
                )
            )
        out[exp_name] = parsed_specs
    return out


def _parse_algorithms(algs: Iterable[dict]) -> List[AlgorithmSpec]:
    parsed: List[AlgorithmSpec] = []
    for a in algs:
        parsed.append(
            AlgorithmSpec(
                name=a.get("name"),
                path=a.get("path"),
                label=a.get("label"),
                meta=a.get("meta", {}),
            )
        )
    return parsed


def _resolve_path(base_path: str, path_template: str, params: Dict[str, Any]) -> str:
    if not path_template:
        return base_path
    rel = path_template.format(**params)
    if base_path:
        return os.path.join(base_path, rel)
    return rel


def _format_label(legend_name: Optional[str], legend_val: Any, cfg: Dict[str, Any], label_template: Optional[str]) -> str:
    if legend_name is not None:
        try:
            return str(legend_name).format(legend_val)
        except Exception:
            return f"{legend_name}{legend_val}"
    if label_template:
        try:
            return label_template.format(**cfg)
        except Exception:
            return str(label_template)
    return str(legend_val)


def _find_config(
    configs: List[Dict[str, Any]],
    exp_name: str,
    iterator_var: Optional[str],
    iterator_val: Any,
    legend_var: str,
    legend_val: Any,
    extra_vars: List[str],
    extra_values: List[Any],
) -> Optional[Dict[str, Any]]:
    for c in configs:
        if c.get("experiment_name") != exp_name:
            continue
        if iterator_var is not None and c.get(iterator_var) != iterator_val:
            continue
        if legend_var is not None and c.get(legend_var) != legend_val:
            continue
        if extra_vars and extra_values and len(extra_vars) == len(extra_values):
            ok = True
            for ev, vv in zip(extra_vars, extra_values):
                if c.get(ev) != vv:
                    ok = False
                    break
            if not ok:
                continue
        return c
    return None


def _filter_configs(
    configs: List[Dict[str, Any]],
    exp_name: str,
    extra_vars: List[str],
    extra_values: List[Any],
    required_vars: List[str],
) -> List[Dict[str, Any]]:
    filtered = []
    for c in configs:
        if c.get("experiment_name") != exp_name:
            continue
        if any(v not in c for v in required_vars):
            continue
        if extra_vars and extra_values and len(extra_vars) == len(extra_values):
            ok = True
            for ev, vv in zip(extra_vars, extra_values):
                if c.get(ev) != vv:
                    ok = False
                    break
            if not ok:
                continue
        filtered.append(c)
    return filtered


def _unique_values(
    configs: List[Dict[str, Any]], row_var: str, col_var: str
) -> Tuple[List[Any], List[Any]]:
    row_values: List[Any] = []
    col_values: List[Any] = []
    seen_r = set()
    seen_c = set()
    for c in configs:
        rv = c.get(row_var)
        cv = c.get(col_var)
        if rv not in seen_r:
            row_values.append(rv)
            seen_r.add(rv)
        if cv not in seen_c:
            col_values.append(cv)
            seen_c.add(cv)
    return row_values, col_values
