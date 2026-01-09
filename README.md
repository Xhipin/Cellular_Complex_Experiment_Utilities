# Cellular Complex Utilities

This repository provides utilities for plotting and metric/statistics aggregation for cellular complex experiments.
It is designed to be generic and decoupled from experiment code. Experiments are expected to write metric artifacts
in a standardized format that this repo can consume.

## Core Flow

1. Read configuration (YAML).
2. Load metric artifacts from explicit run paths.
3. Render plots and tables.

## Metric Artifact Conventions

Artifacts are stored as pickled dictionaries. Two file types are expected:

- `errors_{n}.pkl` for each trial `n`.
- `full_errors.pkl` for the average across trials.

Each dictionary stores metric arrays keyed by the metric name. Conventions:

- **Per-trial metrics** use a `single` suffix: `metric_name + "single"`.
  Example: `tvNMSEsingle`, `pd_curvesingle`.
- **Averaged metrics** use the base name only.
  Example: `tvNMSE`, `pd_curve`.

The plotter will:
- Read per-trial metrics from `errors_{n}.pkl` to compute uncertainty bands.
- Read averaged metrics from `full_errors.pkl` to plot mean curves.

## YAML Configuration (Templates)

This repo expects two YAML files depending on the task:

1. **Ablation (synthetic sweeps)**: one algorithm, parameter sweeps.
2. **Algorithm comparison (real data)**: multiple algorithms, fixed settings per algorithm.

### Ablation YAML (synthetic sweeps)

```yaml
algorithm: RFHORSO
base_path: "/abs/path/to/exp_output"
path_template: "nl_{nl}/std_{std}"
label_template: "nl={nl}, std={std}"

sweeps:
  bandwidth:
    sweep_type: cartesian
    vars: [std, nl]
    values:
      - [0.01, 0.04, 0.1]
      - [2, 4]

plot_sweeps:
  bandwidth:
    - iterator: nl
      legend: std
      legend_name: "$\\nu = {}$"
      iterator_values: [2, 4]
      legend_values: [0.01, 0.04, 0.1]

table_sweeps:
  bandwidth:
    - column_var: std
      row_var: nl
      column_name: "$\\nu = {}$"
      row_name: "$p_{}$"
```

### Algorithm Comparison YAML (real data)

```yaml
algorithms:
  - name: RFHORSO
    path: "/abs/path/to/rfh_output"
  - name: NOGD
    path: "/abs/path/to/nogd_output"

plot_sweeps:
  estimation:
    - iterator: null
      legend: alg
      legend_name: "{}"
      iterator_values: null
      legend_values: [RFHORSO, NOGD]

table_sweeps:
  estimation:
    - column_var: alg
      row_var: metric
      column_name: "{}"
      row_name: "Metric"
```

## Notes

- Ablation uses parameter sweeps plus plot/table sweep specs.
- Algorithm comparison uses plot/table sweeps with `legend: alg`.
- Output paths are provided explicitly in YAML (`base_path` + `path_template` for sweeps, or per-alg `path`).
- The exact set of available metrics is defined by the metric utilities in this repo.

## ROC/AUC (Usage)

ROC curves use `pd_curve` / `pfa_curve` metrics (and their `*single` variants for per-trial data).
AUC is computed by integrating PD vs PFA after tail-averaging and alignment.

Example:

```python
from cellexp_util.plotter.plotter import Plotter, TableMaker
from cellexp_util.plotter.implementations.roc_auc_plotter import (
    render_roc_plot,
    render_auc_table,
    ROCPlotConfig,
    AUCTableConfig,
)

plotter = Plotter("ablation.yaml")
for job_data in plotter.iter_loaded():
    render_roc_plot(job_data, config=ROCPlotConfig(output_dir="experiments/plots"))

tables = TableMaker("ablation.yaml")
for job_data in tables.iter_loaded():
    render_auc_table(job_data, config=AUCTableConfig(output_dir="experiments/plots"))
```

## TODO: Add cellular complex plotting and ROC/AUC utility
