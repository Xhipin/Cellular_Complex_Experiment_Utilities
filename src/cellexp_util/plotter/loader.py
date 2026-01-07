from __future__ import annotations

from dataclasses import dataclass, field
import os
import pickle
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class RunArtifacts:
    base_path: str
    full_errors: Optional[Dict] = None
    trial_errors: List[Dict] = field(default_factory=list)
    trial_paths: List[str] = field(default_factory=list)


class ArtifactLoader:
    def __init__(
        self,
        *,
        full_name: str = "full_errors.pkl",
        trial_prefix: str = "errors_",
        trial_suffix: str = ".pkl",
    ):
        self.full_name = full_name
        self.trial_prefix = trial_prefix
        self.trial_suffix = trial_suffix

    def load_run(
        self,
        path: str,
        *,
        trial_indices: Optional[Sequence[int]] = None,
        max_trials: Optional[int] = None,
    ) -> RunArtifacts:
        base_dir = self._resolve_dir(path)
        full = self.load_full(base_dir)
        trials, trial_paths = self.load_trials(
            base_dir,
            trial_indices=trial_indices,
            max_trials=max_trials,
        )
        return RunArtifacts(base_path=base_dir, full_errors=full, trial_errors=trials, trial_paths=trial_paths)

    def load_full(self, path: str) -> Optional[Dict]:
        base_dir = self._resolve_dir(path)
        full_path = os.path.join(base_dir, self.full_name)
        return self._load_pickle(full_path)

    def load_trials(
        self,
        path: str,
        *,
        trial_indices: Optional[Sequence[int]] = None,
        max_trials: Optional[int] = None,
    ) -> Tuple[List[Dict], List[str]]:
        base_dir = self._resolve_dir(path)
        trial_paths: List[str] = []
        if trial_indices is not None:
            for idx in trial_indices:
                trial_paths.append(self._trial_path(base_dir, idx))
        elif max_trials is not None:
            for idx in range(int(max_trials)):
                trial_paths.append(self._trial_path(base_dir, idx))
        else:
            trial_paths = self._glob_trial_paths(base_dir)

        trials: List[Dict] = []
        kept_paths: List[str] = []
        for p in trial_paths:
            data = self._load_pickle(p)
            if data is None:
                continue
            trials.append(data)
            kept_paths.append(p)
        return trials, kept_paths

    def _resolve_dir(self, path: str) -> str:
        if os.path.isdir(path):
            return path
        if os.path.isfile(path):
            return os.path.dirname(path)
        return path

    def _trial_path(self, base_dir: str, idx: int) -> str:
        return os.path.join(base_dir, f"{self.trial_prefix}{idx}{self.trial_suffix}")

    def _glob_trial_paths(self, base_dir: str) -> List[str]:
        if not os.path.isdir(base_dir):
            return []
        candidates = []
        for name in os.listdir(base_dir):
            if not name.startswith(self.trial_prefix) or not name.endswith(self.trial_suffix):
                continue
            candidates.append(os.path.join(base_dir, name))
        candidates.sort(key=self._trial_sort_key)
        return candidates

    def _trial_sort_key(self, path: str) -> Tuple[int, str]:
        name = os.path.basename(path)
        prefix_len = len(self.trial_prefix)
        suffix_len = len(self.trial_suffix)
        core = name[prefix_len:len(name) - suffix_len]
        try:
            return (int(core), name)
        except Exception:
            return (1 << 30, name)

    def _load_pickle(self, path: str) -> Optional[Dict]:
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            return data if isinstance(data, dict) else None
        except Exception:
            return None


def load_runs(
    paths: Iterable[str],
    *,
    trial_indices: Optional[Sequence[int]] = None,
    max_trials: Optional[int] = None,
) -> List[RunArtifacts]:
    loader = ArtifactLoader()
    out: List[RunArtifacts] = []
    for p in paths:
        out.append(loader.load_run(p, trial_indices=trial_indices, max_trials=max_trials))
    return out
