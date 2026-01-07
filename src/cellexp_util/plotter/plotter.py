from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .config_utils import (
    AblationManifest,
    ComparisonManifest,
    PlotJob,
    ResolvedCase,
    TableJob,
    load_manifest,
)
from .loader import ArtifactLoader, RunArtifacts

Manifest = AblationManifest | ComparisonManifest


@dataclass
class PlotSeriesData:
    case: ResolvedCase
    artifacts: RunArtifacts


@dataclass
class PlotJobData:
    job: PlotJob
    series: List[PlotSeriesData]


@dataclass
class TableCellData:
    cases: List[ResolvedCase]
    artifacts: List[RunArtifacts]


@dataclass
class TableJobData:
    job: TableJob
    cells: Dict[Tuple[object, object], TableCellData]
    algs: Dict[str, RunArtifacts]


class Plotter:
    def __init__(
        self,
        yaml_path: str,
        *,
        loader: Optional[ArtifactLoader] = None,
        trial_indices: Optional[Iterable[int]] = None,
        max_trials: Optional[int] = None,
    ):
        self.manifest: Manifest = load_manifest(yaml_path)
        self.loader = loader or ArtifactLoader()
        self.trial_indices = list(trial_indices) if trial_indices is not None else None
        self.max_trials = max_trials

    def list_jobs(self) -> List[PlotJob]:
        return self.manifest.plot_jobs()

    def load_job(self, job: PlotJob) -> PlotJobData:
        series: List[PlotSeriesData] = []
        for case in job.curves:
            artifacts = self.loader.load_run(
                case.path,
                trial_indices=self.trial_indices,
                max_trials=self.max_trials,
            )
            series.append(PlotSeriesData(case=case, artifacts=artifacts))
        return PlotJobData(job=job, series=series)

    def iter_loaded(self) -> Iterable[PlotJobData]:
        for job in self.list_jobs():
            yield self.load_job(job)


class TableMaker:
    def __init__(
        self,
        yaml_path: str,
        *,
        loader: Optional[ArtifactLoader] = None,
        trial_indices: Optional[Iterable[int]] = None,
        max_trials: Optional[int] = None,
    ):
        self.manifest: Manifest = load_manifest(yaml_path)
        self.loader = loader or ArtifactLoader()
        self.trial_indices = list(trial_indices) if trial_indices is not None else None
        self.max_trials = max_trials

    def list_jobs(self) -> List[TableJob]:
        return self.manifest.table_jobs()

    def load_job(self, job: TableJob) -> TableJobData:
        cells: Dict[Tuple[object, object], TableCellData] = {}
        algs: Dict[str, RunArtifacts] = {}

        if job.cells:
            for key, cases in job.cells.items():
                artifacts = [
                    self.loader.load_run(
                        c.path,
                        trial_indices=self.trial_indices,
                        max_trials=self.max_trials,
                    )
                    for c in cases
                ]
                cells[key] = TableCellData(cases=list(cases), artifacts=artifacts)

        if not cells and job.alg_paths:
            for alg, path in job.alg_paths.items():
                algs[alg] = self.loader.load_run(
                    path,
                    trial_indices=self.trial_indices,
                    max_trials=self.max_trials,
                )

        return TableJobData(job=job, cells=cells, algs=algs)

    def iter_loaded(self) -> Iterable[TableJobData]:
        for job in self.list_jobs():
            yield self.load_job(job)
