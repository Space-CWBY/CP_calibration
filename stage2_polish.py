# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Standalone Stage-2 polishing driver (distance-lock in standardized CLR space).

This standalone script was prepared for public release.
- No hard-coded local machine paths are included.
- f0 = physics loss
- f1 = prior penalty
- Raw experimental data are not bundled in this release.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "5")
os.environ.setdefault("MKL_NUM_THREADS", "5")


# ===== BEGIN INLINED MODULE: constants.py =====


from pathlib import Path

CASES = ["ND_rot", "RD_rot", "TD_rot", "RD_ten"]
ANCHORS = [0.02, 0.06, 0.10]
MECH_ORDER = ["basal", "prismatic", "pyramidal", "twin_tensile"]
KEY_ORDER = ["tau0", "tau1", "thet0", "thet1"]
MECH_LABELS = ["Basal", "Prismatic", "Pyramidal", "Tensile twin"]
CASE_TO_EXPERIMENT_FILE = {
    "ND_rot": "ND_compression_231102.csv",
    "RD_rot": "RD_compression_231102.csv",
    "TD_rot": "TD_compression_231102.csv",
    "RD_ten": "RD_tension.csv",
}
CASE_TO_TEMPLATE_FILE = {
    "ND_rot": "vpsc7_ND_rot.in",
    "RD_rot": "vpsc7_RD_rot.in",
    "TD_rot": "vpsc7_TD_rot.in",
    "RD_ten": "vpsc7_RD_ten.in",
}
SX_TEMPLATE_NAME = "sx"

BOUNDS = {
    "basal": {"tau0": (5.0, 300.0), "tau1": (0.0, 150.0), "thet0": (0.0, 1000.0), "thet1": (0.0, 300.0)},
    "prismatic": {"tau0": (30.0, 500.0), "tau1": (0.0, 150.0), "thet0": (0.0, 2500.0), "thet1": (0.0, 200.0)},
    "pyramidal": {"tau0": (30.0, 500.0), "tau1": (0.0, 1000.0), "thet0": (0.0, 10000.0), "thet1": (0.0, 2000.0)},
    "twin_tensile": {"tau0": (5.0, 500.0), "tau1": (0.0, 50.0), "thet0": (0.0, 1000.0), "thet1": (0.0, 1000.0)},
}

OMP_NUM_THREADS = "5"
MKL_NUM_THREADS = "5"

DEFAULT_SIGMA_SCALE_EPS = 0.06
DEFAULT_EPS_LIN = 0.03
DEFAULT_SMOOTH_STRAIN = 0.01
DEFAULT_SMOOTH_MIN_WIN_PTS = 9
DEFAULT_N_GRID_TAIL = 260
DEFAULT_FAIL_PENALTY = 1.0e6

def anchor_tag(anchor: float) -> str:
    return f"e{float(anchor):.2f}".replace(".", "p")


def required_actrel_columns() -> list[str]:
    cols: list[str] = []
    for case in CASES:
        for anchor in ANCHORS:
            tag = anchor_tag(anchor)
            for mech in MECH_ORDER:
                cols.append(f"act_rel_{case}_{mech}_{tag}")
    return cols


def voce_column_names() -> list[str]:
    return [f"{mech}_{key}" for mech in MECH_ORDER for key in KEY_ORDER]

# ===== END INLINED MODULE: constants.py =====


# ===== BEGIN INLINED MODULE: types.py =====


from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuntimePaths:
    evpsc_root: Path
    evpsc_exe: Path
    exp_dir: Path
    template_dir: Path
    scratch_dir: Path


@dataclass(frozen=True)
class Stage1Options:
    timeout_sec: int | None = 120
    parallel_cases: int = 4
    w_rmse: float = 1.0
    w_dshape: float = 1.0
    export_curves: bool = False


@dataclass(frozen=True)
class ClusteringOptions:
    prior_feasible: float = 0.05
    use_prior_feasible: bool = True
    top_frac: float = 0.10
    k_min: int = 2
    k_max: int = 12
    elbow_plus: int = 2
    elbow_smooth: int = 3
    random_state: int = 0
    n_neighbors: int = 40
    min_dist: float = 0.05
    metric: str = "euclidean"
    n_boot: int = 30
    subsample_frac: float = 0.85


@dataclass(frozen=True)
class PolishingOptions:
    timeout_sec: int = 120
    parallel_cases: int = 4
    w_rmse: float = 1.0
    w_dshape: float = 1.0
    lam_c: float = 5.0
    nmax: int = 200
    stall_evals: int = 50
    sigma0: float = 0.08
    d2n_gate_q: float = 0.95
    seed: int = 0
    random_state: int = 0
    rep_types: tuple[str, ...] = ("best_balanced",)

# ===== END INLINED MODULE: types.py =====


# ===== BEGIN INLINED MODULE: io_utils.py =====


import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd



def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, encoding="cp949")
    except Exception:
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except UnicodeDecodeError:
            return pd.read_csv(path, engine="python", on_bad_lines="skip", encoding="cp949")


def read_whitespace_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", comment="#", engine="python")


def read_numeric_two_column_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    rows: list[list[float]] = []
    for line in lines:
        text = line.strip()
        if not text or text.startswith("#") or text.startswith("//"):
            continue
        tokens = re.split(r"[,\t; ]+", text)
        values: list[float] = []
        for token in tokens:
            try:
                values.append(float(token))
            except Exception:
                continue
        if len(values) >= 2:
            rows.append(values[:2])
    if len(rows) < 5:
        raise ValueError(f"Not enough numeric rows in file: {path}")
    arr = np.asarray(rows, dtype=float)
    x = arr[:, 0].copy()
    y = arr[:, 1].copy()
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if x[-1] < 0:
        x = -x
        y = -y
        order = np.argsort(x)
        x = x[order]
        y = y[order]
    if np.nanmedian(y[: max(5, len(y) // 10)]) < 0 and x[-1] > 0:
        y = -y
    xu, idx = np.unique(x, return_index=True)
    yu = y[idx]
    return xu, yu


def parse_vpsc7_paths(vpsc7_text: str) -> dict[str, str]:
    lines = vpsc7_text.splitlines()
    texture = ""
    sx = ""
    process = ""

    def next_nonempty(start: int) -> tuple[int, str]:
        j = start
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j >= len(lines):
            return j, ""
        return j, lines[j].strip()

    for i, line in enumerate(lines):
        if "filetext" in line.lower():
            _, texture = next_nonempty(i + 1)
            break
    for i, line in enumerate(lines):
        if "filecrys" in line.lower():
            _, sx = next_nonempty(i + 1)
            break
    for i, line in enumerate(lines):
        if "list of processes" in line.lower():
            for j in range(i + 1, min(len(lines), i + 80)):
                text = lines[j].strip()
                if not text or text.startswith("*"):
                    continue
                if ".pro" in text.lower():
                    process = text
                    break
            break
    if not process:
        for line in lines:
            if ".pro" in line.lower():
                process = line.strip()
                break
    return {"texture": texture.strip(), "sx": sx.strip(), "process": process.strip()}


def rewrite_vpsc7_paths(vpsc7_text: str, texture_path: str, sx_path: str, process_path: str) -> str:
    lines = vpsc7_text.splitlines()
    out = lines[:]

    def next_nonempty(start: int) -> tuple[int, str]:
        j = start
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j >= len(lines):
            return j, ""
        return j, lines[j].strip()

    for i, line in enumerate(lines):
        if "filetext" in line.lower():
            j, _ = next_nonempty(i + 1)
            if j < len(out):
                out[j] = texture_path
            break

    for i, line in enumerate(lines):
        if "filecrys" in line.lower():
            j, _ = next_nonempty(i + 1)
            if j < len(out):
                out[j] = sx_path
            break

    replaced = False
    for i, line in enumerate(lines):
        if "list of processes" in line.lower():
            for j in range(i + 1, min(len(lines), i + 80)):
                text = lines[j].strip()
                if not text or text.startswith("*"):
                    continue
                if ".pro" in text.lower():
                    out[j] = process_path
                    replaced = True
                    break
            break
    if not replaced:
        for j, line in enumerate(lines):
            if ".pro" in line.lower():
                out[j] = process_path
                break
    return "\n".join(out) + "\n"


def make_stage1_schema() -> list[str]:
    columns = [
        "trial",
        "state",
        "f0",
        "f1",
        "prior_valid",
        "f0_rmse_mean_raw",
        "f0_dshape_mean_raw",
        "f0_rmse_mean",
        "f0_dshape_mean",
    ]
    for case in CASES:
        columns.extend(
            [
                f"f0_rmse_{case}_raw",
                f"f0_dshape_{case}_raw",
                f"f0_case_{case}",
                f"r2_{case}",
            ]
        )
    for mech in MECH_ORDER:
        for key in KEY_ORDER:
            columns.append(f"{mech}_{key}")
    for case in CASES:
        for anchor in ANCHORS:
            tag = anchor_tag(anchor)
            for mech in MECH_ORDER:
                columns.append(f"act_raw_{case}_{mech}_{tag}")
            for mech in MECH_ORDER:
                columns.append(f"act_rel_{case}_{mech}_{tag}")
            columns.append(f"twfr4_{case}_{tag}")
    return columns


STAGE1_SCHEMA = make_stage1_schema()


def _read_existing_trials_max_trial(trials_csv: Path) -> int:
    if not trials_csv.exists():
        return -1
    try:
        with trials_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "trial" not in reader.fieldnames:
                return -1
            maximum = -1
            for row in reader:
                try:
                    value = int(float(row.get("trial", -1)))
                    maximum = max(maximum, value)
                except Exception:
                    continue
            return maximum
    except Exception:
        return -1


def next_trial_id(trials_csv: Path) -> int:
    return _read_existing_trials_max_trial(trials_csv) + 1


def append_row_fixed_atomic(csv_path: Path, row: dict[str, Any], schema: Iterable[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    schema_list = list(schema)
    fixed = {key: row.get(key, np.nan) for key in schema_list}
    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")

    if not csv_path.exists():
        with tmp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=schema_list)
            writer.writeheader()
            writer.writerow(fixed)
        tmp_path.replace(csv_path)
        return

    with csv_path.open("r", encoding="utf-8", newline="") as old_handle, tmp_path.open("w", encoding="utf-8", newline="") as new_handle:
        reader = csv.DictReader(old_handle)
        writer = csv.DictWriter(new_handle, fieldnames=schema_list)
        writer.writeheader()
        for old_row in reader:
            writer.writerow({key: old_row.get(key, np.nan) for key in schema_list})
        writer.writerow(fixed)
    tmp_path.replace(csv_path)


def append_jsonl_row(path: Path, row: dict[str, Any], schema: Iterable[str]) -> None:
    fixed = {key: row.get(key, np.nan) for key in schema}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(fixed, ensure_ascii=False) + "\n")

# ===== END INLINED MODULE: io_utils.py =====


# ===== BEGIN INLINED MODULE: metrics.py =====


import numpy as np


def rmse_interp(sim_eps: np.ndarray, sim_sig: np.ndarray, exp_eps: np.ndarray, exp_sig: np.ndarray) -> float:
    eps_min = max(float(np.min(sim_eps)), float(np.min(exp_eps)))
    eps_max = min(float(np.max(sim_eps)), float(np.max(exp_eps)))
    mask = (exp_eps >= eps_min) & (exp_eps <= eps_max)
    if np.sum(mask) < 5:
        return float("inf")
    x = exp_eps[mask]
    y = exp_sig[mask]
    yhat = np.interp(x, sim_eps, sim_sig)
    return float(np.sqrt(np.mean((yhat - y) ** 2)))


def r2_interp(sim_eps: np.ndarray, sim_sig: np.ndarray, exp_eps: np.ndarray, exp_sig: np.ndarray) -> float:
    eps_min = max(float(np.min(sim_eps)), float(np.min(exp_eps)))
    eps_max = min(float(np.max(sim_eps)), float(np.max(exp_eps)))
    mask = (exp_eps >= eps_min) & (exp_eps <= eps_max)
    if np.sum(mask) < 5:
        return float("nan")
    x = exp_eps[mask]
    y = exp_sig[mask]
    yhat = np.interp(x, sim_eps, sim_sig)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def sigma_at(strain: float, eps: np.ndarray, sig: np.ndarray) -> float:
    if strain <= float(eps[0]):
        return float(sig[0])
    if strain >= float(eps[-1]):
        return float(sig[-1])
    return float(np.interp(float(strain), eps, sig))


def drmse_scaled(sim_eps: np.ndarray, sim_sig: np.ndarray, exp_eps: np.ndarray, exp_sig: np.ndarray, sigma_scale: float) -> float:
    eps_min = max(float(np.min(sim_eps)), float(np.min(exp_eps)))
    eps_max = min(float(np.max(sim_eps)), float(np.max(exp_eps)))
    mask = (exp_eps >= eps_min) & (exp_eps <= eps_max)
    if np.sum(mask) < 8:
        return float("inf")
    x = exp_eps[mask]
    y = exp_sig[mask]
    yhat = np.interp(x, sim_eps, sim_sig)
    dy = np.gradient(y, x)
    dyhat = np.gradient(yhat, x)
    scale = float(abs(sigma_scale))
    if scale < 1e-6:
        scale = max(1e-6, float(np.nanmax(np.abs(y))))
    return float(np.sqrt(np.mean(((dyhat - dy) / scale) ** 2)))

# ===== END INLINED MODULE: metrics.py =====


# ===== BEGIN INLINED MODULE: voce.py =====


import math
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np



@dataclass(frozen=True)
class Voce4:
    tau0: float
    tau1: float
    thet0: float
    thet1: float


@dataclass(frozen=True)
class VoceParams:
    basal: Voce4
    prismatic: Voce4
    pyramidal: Voce4
    twin_tensile: Voce4

    @staticmethod
    def from_vector(x16: Iterable[float]) -> "VoceParams":
        x = [float(v) for v in x16]
        if len(x) != 16:
            raise ValueError(f"x16 must have length 16, got {len(x)}")
        return VoceParams(
            basal=Voce4(*x[0:4]),
            prismatic=Voce4(*x[4:8]),
            pyramidal=Voce4(*x[8:12]),
            twin_tensile=Voce4(*x[12:16]),
        )

    def to_flat_dict(self) -> dict[str, float]:
        values = {
            "basal": self.basal,
            "prismatic": self.prismatic,
            "pyramidal": self.pyramidal,
            "twin_tensile": self.twin_tensile,
        }
        out: dict[str, float] = {}
        for mech, voce in values.items():
            out[f"{mech}_tau0"] = float(voce.tau0)
            out[f"{mech}_tau1"] = float(voce.tau1)
            out[f"{mech}_thet0"] = float(voce.thet0)
            out[f"{mech}_thet1"] = float(voce.thet1)
        return out


@dataclass(frozen=True)
class VarSpec:
    mech: str
    key: str
    lo: float
    hi: float
    scale: str


def build_var_specs(theta_scale: str = "log") -> list[VarSpec]:
    specs: list[VarSpec] = []
    for mech in MECH_ORDER:
        for key in KEY_ORDER:
            lo, hi = BOUNDS[mech][key]
            scale = theta_scale if key.startswith("thet") else "lin"
            specs.append(VarSpec(mech, key, float(lo), float(hi), scale))
    return specs


_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _replace_first4_floats_keep_tail(line: str, voce: Voce4) -> str:
    numbers = list(_FLOAT_RE.finditer(line))
    if len(numbers) < 4:
        raise ValueError("Cannot replace Voce floats: found fewer than four numbers.")
    values = [voce.tau0, voce.tau1, voce.thet0, voce.thet1]
    parts: list[str] = []
    last = 0
    for idx in range(4):
        match = numbers[idx]
        parts.append(line[last:match.start()])
        parts.append(f"{values[idx]:.10g}")
        last = match.end()
    parts.append(line[last:])
    return "".join(parts)


def update_sx_voce_robust(sx_text: str, params: VoceParams) -> str:
    lines = sx_text.splitlines()
    targets = [
        ("BASAL <a> SLIP", params.basal, ["BASAL", "BASAL <A>"]),
        ("PRISMATIC <a> SLIP", params.prismatic, ["PRISMATIC", "PRISMATIC <A>"]),
        ("PYRAMIDAL <c+a> SLIP", params.pyramidal, ["PYRAMIDAL", "C+A", "C+A SLIP"]),
        ("{10-12} TENSILE TWIN", params.twin_tensile, ["TWIN", "TENSILE TWIN", "10-12"]),
    ]

    def find_header(primary: str, fallbacks: list[str]) -> int:
        p = primary.lower()
        for i, line in enumerate(lines):
            if p in line.lower():
                return i
        for i, line in enumerate(lines):
            hit = sum(1 for token in fallbacks if token.lower() in line.lower())
            if hit >= max(1, len(fallbacks) // 2):
                return i
        raise ValueError(f"Cannot find sx section header: {primary}")

    def find_param_line(header_index: int) -> int:
        for j in range(header_index + 1, min(len(lines), header_index + 120)):
            if len(_FLOAT_RE.findall(lines[j])) >= 4:
                return j
        raise ValueError("Cannot locate Voce parameter line after header.")

    for header, voce, fallbacks in targets:
        i_header = find_header(header, fallbacks)
        i_param = find_param_line(i_header)
        lines[i_param] = _replace_first4_floats_keep_tail(lines[i_param], voce)

    return "\n".join(lines) + "\n"


def _log_map(u: float, lo: float, hi: float, eps: float) -> float:
    lo2 = max(float(lo), float(eps))
    hi2 = max(float(hi), lo2)
    return 10.0 ** (math.log10(lo2) + float(u) * (math.log10(hi2) - math.log10(lo2)))


def unit_to_x16(u16: np.ndarray, specs: list[VarSpec], eps_theta: float = 1e-6) -> list[float]:
    u = np.asarray(u16, dtype=float).reshape(-1)
    if u.size != 16:
        raise ValueError("u16 must have 16 entries")
    x = np.zeros(16, dtype=float)
    for i, spec in enumerate(specs):
        ui = float(np.clip(u[i], 0.0, 1.0))
        if spec.scale == "lin":
            x[i] = spec.lo + ui * (spec.hi - spec.lo)
        elif spec.scale == "log":
            x[i] = 0.0 if spec.hi <= 0.0 else _log_map(ui, spec.lo, spec.hi, eps_theta)
        else:
            raise ValueError(f"Unknown scale: {spec.scale}")
    return [float(v) for v in x.tolist()]


def x16_to_unit(x16: Iterable[float], specs: list[VarSpec], eps_theta: float = 1e-6) -> np.ndarray:
    x = [float(v) for v in x16]
    u = np.zeros(16, dtype=float)
    for i, spec in enumerate(specs):
        value = x[i]
        if spec.scale == "lin":
            den = spec.hi - spec.lo
            u[i] = 0.0 if abs(den) < 1e-12 else float(np.clip((value - spec.lo) / den, 0.0, 1.0))
        else:
            lo2 = max(spec.lo, eps_theta)
            hi2 = max(spec.hi, lo2)
            if value <= lo2 or hi2 <= lo2:
                u[i] = 0.0
            else:
                a = math.log10(lo2)
                b = math.log10(hi2)
                u[i] = float(np.clip((math.log10(max(value, lo2)) - a) / max(b - a, 1e-12), 0.0, 1.0))
    return u


def mirror01(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float).copy()
    x = np.where(x < 0.0, -x, x)
    x = np.where(x > 1.0, 2.0 - x, x)
    x = np.where(x < 0.0, 0.0, x)
    x = np.where(x > 1.0, 1.0, x)
    return x

# ===== END INLINED MODULE: voce.py =====


# ===== BEGIN INLINED MODULE: activity.py =====


import numpy as np



def _moving_average_tail(values: np.ndarray, start: int, win: int) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    out = x.copy()
    if win < 3 or start >= len(x) - 2:
        return out
    tail = x[start:]
    pad = win // 2
    tail_pad = np.pad(tail, (pad, pad), mode="edge")
    kernel = np.ones(win, dtype=float) / float(win)
    out[start:] = np.convolve(tail_pad, kernel, mode="valid")
    return out


def modes_to_fractions(modes: np.ndarray) -> np.ndarray:
    x = np.asarray(modes, dtype=float)
    if x.ndim != 2:
        raise ValueError("modes must be 2D")
    out = np.zeros((x.shape[0], 4), dtype=float)
    out[:, : min(4, x.shape[1])] = x[:, : min(4, x.shape[1])]
    out = np.clip(out, 0.0, None)
    sums = np.sum(out, axis=1)
    sums[sums <= 1e-12] = 1.0
    return out / sums[:, None]


def preprocess_modes_piecewise_raw(
    eps_raw: np.ndarray,
    modes_raw: np.ndarray | None,
    twfr_raw: np.ndarray | None = None,
    *,
    eps_lin: float = DEFAULT_EPS_LIN,
    n_grid_tail: int = DEFAULT_N_GRID_TAIL,
    smooth_strain: float = DEFAULT_SMOOTH_STRAIN,
    smooth_min_win_pts: int = DEFAULT_SMOOTH_MIN_WIN_PTS,
) -> dict[str, np.ndarray]:
    eps = np.asarray(eps_raw, dtype=float)
    order = np.argsort(eps)
    eps = eps[order]

    if modes_raw is None or np.asarray(modes_raw).size == 0:
        modes4 = np.zeros((len(eps), 4), dtype=float)
    else:
        mr = np.asarray(modes_raw, dtype=float)[order]
        modes4 = np.zeros((mr.shape[0], 4), dtype=float)
        if mr.ndim == 2:
            modes4[:, : min(4, mr.shape[1])] = mr[:, : min(4, mr.shape[1])]
    modes4 = np.clip(modes4, 0.0, None)

    if twfr_raw is None or np.asarray(twfr_raw).size == 0:
        twfr = np.full(len(eps), np.nan, dtype=float)
    else:
        twfr = np.asarray(twfr_raw, dtype=float)[order]

    eps_max = float(np.max(eps)) if len(eps) else 0.0
    eps_lin = float(min(eps_lin, eps_max))
    n_dense = max(50, int(eps_lin / 0.00025) + 1) if eps_lin > 0 else 50
    eps_dense = np.linspace(0.0, eps_lin, n_dense)
    if eps_max > eps_lin + 1e-9:
        eps_tail = np.linspace(eps_lin, eps_max, max(20, int(n_grid_tail)))
        eps_grid = np.concatenate([eps_dense[:-1], eps_tail])
    else:
        eps_grid = eps_dense

    mode_grid = np.zeros((len(eps_grid), 4), dtype=float)
    for j in range(4):
        mode_grid[:, j] = np.interp(eps_grid, eps, modes4[:, j])

    if np.any(np.isfinite(twfr)):
        twfr_grid = np.interp(eps_grid, eps, twfr)
    else:
        twfr_grid = np.full(len(eps_grid), np.nan, dtype=float)

    if eps_max > eps_lin + 1e-9 and len(eps_grid) >= 5:
        start = int(np.argmax(eps_grid >= eps_lin))
        tail_x = eps_grid[start:]
        if len(tail_x) >= 5:
            dx = float(np.median(np.diff(tail_x)))
            win = max(smooth_min_win_pts, int(round(smooth_strain / max(dx, 1e-9))))
            if win % 2 == 0:
                win += 1
            for j in range(4):
                mode_grid[:, j] = _moving_average_tail(mode_grid[:, j], start, win)
            if np.any(np.isfinite(twfr_grid)):
                twfr_grid = _moving_average_tail(twfr_grid, start, win)

    return {
        "eps": eps_grid,
        "mode_raw": np.clip(mode_grid, 0.0, None),
        "twfr": np.clip(twfr_grid, 0.0, 1.0) if np.any(np.isfinite(twfr_grid)) else twfr_grid,
    }


def relative_fractions_at_anchors_from_raw(eps_proc: np.ndarray, mode_raw_proc: np.ndarray, anchors: list[float]) -> np.ndarray:
    eps = np.asarray(eps_proc, dtype=float)
    mode = np.asarray(mode_raw_proc, dtype=float)
    out = np.full((len(anchors), 4), np.nan, dtype=float)
    for i, anchor in enumerate(anchors):
        values = []
        for j in range(4):
            if float(anchor) < float(eps[0]) or float(anchor) > float(eps[-1]):
                values.append(float("nan"))
            else:
                values.append(float(np.interp(float(anchor), eps, mode[:, j])))
        v = np.asarray(values, dtype=float)
        v = np.clip(v, 0.0, None)
        s = float(np.nansum(v))
        if s <= 1e-12:
            continue
        out[i, :] = v / s
    return out


def sample_at_anchors(x: np.ndarray, y: np.ndarray, anchors: list[float]) -> list[float]:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    values: list[float] = []
    for anchor in anchors:
        if float(anchor) < float(xx[0]) or float(anchor) > float(xx[-1]):
            values.append(float("nan"))
        else:
            values.append(float(np.interp(float(anchor), xx, yy)))
    return values

# ===== END INLINED MODULE: activity.py =====


# ===== BEGIN INLINED MODULE: features.py =====


import numpy as np
import pandas as pd



def build_clr_features_from_df(df: pd.DataFrame, eps: float = 1e-12) -> np.ndarray:
    parts = []
    for case in CASES:
        for anchor in ANCHORS:
            tag = anchor_tag(anchor)
            cols = [f"act_rel_{case}_{mech}_{tag}" for mech in MECH_ORDER]
            block = df[cols].to_numpy(dtype=float)
            block = np.clip(block, eps, None)
            block = block / np.sum(block, axis=1, keepdims=True)
            logb = np.log(block)
            parts.append(logb - logb.mean(axis=1, keepdims=True))
    return np.concatenate(parts, axis=1)


def build_clr_features_from_anchor_dict(anchors: dict[str, float], eps: float = 1e-12) -> np.ndarray:
    parts = []
    for case in CASES:
        for anchor in ANCHORS:
            tag = anchor_tag(anchor)
            block = np.asarray([float(anchors.get(f"act_rel_{case}_{mech}_{tag}", np.nan)) for mech in MECH_ORDER], dtype=float)
            if not np.isfinite(block).all():
                return np.full((len(CASES) * len(ANCHORS) * 4,), np.nan, dtype=float)
            block = np.clip(block, eps, None)
            block = block / float(np.sum(block))
            logb = np.log(block)
            parts.append((logb - float(np.mean(logb))).reshape(4))
    return np.concatenate(parts, axis=0)


def compute_rmse_mean(df: pd.DataFrame) -> pd.Series:
    if "f0_rmse_mean_raw" in df.columns:
        return pd.to_numeric(df["f0_rmse_mean_raw"], errors="coerce")
    cols = [f"f0_rmse_{case}_raw" for case in CASES if f"f0_rmse_{case}_raw" in df.columns]
    if not cols:
        raise RuntimeError("RMSE columns missing.")
    return df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)

# ===== END INLINED MODULE: features.py =====


# ===== BEGIN INLINED MODULE: prior.py =====


import numpy as np



def hinge_pos(x: float) -> float:
    return float(x) if x > 0.0 else 0.0


def _get_p(anchors_dict: dict[str, float], case: str, mech: str, tag: str) -> float:
    value = anchors_dict.get(f"act_rel_{case}_{mech}_{tag}", np.nan)
    try:
        return float(value)
    except Exception:
        return float("nan")


def prior_penalty_v2(anchors_dict: dict[str, float]) -> tuple[float, dict[str, float]]:
    tags_all = [anchor_tag(a) for a in [0.02, 0.06, 0.10]]
    tags_weak = [anchor_tag(a) for a in [0.06, 0.10]]
    lam1, lam2, lam3, lam4, lam5, lam6 = 1.0, 1.0, 0.5, 0.5, 0.2, 0.2

    t1 = t2 = t3 = t4 = 0.0
    for tag in tags_all:
        p_twin_rd = _get_p(anchors_dict, "RD_rot", "twin_tensile", tag)
        p_twin_td = _get_p(anchors_dict, "TD_rot", "twin_tensile", tag)
        p_twin_nd = _get_p(anchors_dict, "ND_rot", "twin_tensile", tag)
        p_twin_ten = _get_p(anchors_dict, "RD_ten", "twin_tensile", tag)
        if not all(np.isfinite(v) for v in [p_twin_rd, p_twin_td, p_twin_nd, p_twin_ten]):
            return float(DEFAULT_FAIL_PENALTY), {f"term{i}": float(DEFAULT_FAIL_PENALTY) for i in range(1, 7)}
        t1 += lam1 * hinge_pos(p_twin_td - p_twin_rd)
        t2 += lam2 * hinge_pos(p_twin_ten - p_twin_rd)
        t3 += lam3 * hinge_pos(p_twin_nd - min(p_twin_rd, p_twin_td))

    for tag in tags_weak:
        p_b_nd = _get_p(anchors_dict, "ND_rot", "basal", tag)
        p_b_rd = _get_p(anchors_dict, "RD_rot", "basal", tag)
        p_b_td = _get_p(anchors_dict, "TD_rot", "basal", tag)
        if not all(np.isfinite(v) for v in [p_b_nd, p_b_rd, p_b_td]):
            return float(DEFAULT_FAIL_PENALTY), {f"term{i}": float(DEFAULT_FAIL_PENALTY) for i in range(1, 7)}
        t4 += lam4 * hinge_pos(max(p_b_rd, p_b_td) - p_b_nd)

    tag06 = anchor_tag(0.06)
    tag10 = anchor_tag(0.10)
    p_pr_rd_06 = _get_p(anchors_dict, "RD_rot", "prismatic", tag06)
    p_pr_rd_10 = _get_p(anchors_dict, "RD_rot", "prismatic", tag10)
    p_pr_td_06 = _get_p(anchors_dict, "TD_rot", "prismatic", tag06)
    p_pr_td_10 = _get_p(anchors_dict, "TD_rot", "prismatic", tag10)
    if not all(np.isfinite(v) for v in [p_pr_rd_06, p_pr_rd_10, p_pr_td_06, p_pr_td_10]):
        return float(DEFAULT_FAIL_PENALTY), {f"term{i}": float(DEFAULT_FAIL_PENALTY) for i in range(1, 7)}

    t5 = lam5 * hinge_pos(p_pr_rd_10 - p_pr_rd_06)
    t6 = lam6 * hinge_pos(p_pr_td_06 - p_pr_td_10)
    f1 = float(t1 + t2 + t3 + t4 + t5 + t6)
    terms = {"term1": float(t1), "term2": float(t2), "term3": float(t3), "term4": float(t4), "term5": float(t5), "term6": float(t6)}
    return f1, terms

# ===== END INLINED MODULE: prior.py =====


# ===== BEGIN INLINED MODULE: evpsc.py =====


import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np



os.environ.setdefault("OMP_NUM_THREADS", OMP_NUM_THREADS)
os.environ.setdefault("MKL_NUM_THREADS", MKL_NUM_THREADS)


def resolve_repo_path(evpsc_root: Path, path_str: str) -> Path:
    text = (path_str or "").strip().strip('"').strip("'")
    if not text:
        return Path("")
    path = Path(text)
    return path if path.is_absolute() else (evpsc_root / path).resolve()


def ensure_evpsc_environment() -> dict[str, str]:
    return os.environ.copy()


def purge_case_dir(case_dir: Path) -> None:
    keep_names = {"vpsc7.in", "sx"}
    for path in case_dir.glob("*"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            continue
        if path.name in keep_names or path.suffix.lower() in {".odf", ".pro"}:
            continue
        try:
            path.unlink()
        except Exception:
            pass


def prepare_case_run_dir(runtime: RuntimePaths, case_name: str, voce: VoceParams) -> Path:
    template_in = runtime.template_dir / CASE_TO_TEMPLATE_FILE[case_name]
    sx_template = runtime.template_dir / SX_TEMPLATE_NAME
    if not template_in.exists():
        raise FileNotFoundError(f"Missing template file: {template_in}")
    if not sx_template.exists():
        raise FileNotFoundError(f"Missing sx template: {sx_template}")

    case_dir = runtime.scratch_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    purge_case_dir(case_dir)

    vpsc7_src = template_in.read_text(encoding="utf-8", errors="replace")
    paths = parse_vpsc7_paths(vpsc7_src)
    texture_src = resolve_repo_path(runtime.evpsc_root, paths["texture"])
    process_src = resolve_repo_path(runtime.evpsc_root, paths["process"])
    if not texture_src.exists():
        raise FileNotFoundError(f"Texture file not found: {texture_src}")
    if not process_src.exists():
        raise FileNotFoundError(f"Process file not found: {process_src}")

    texture_dst = case_dir / texture_src.name
    process_dst = case_dir / process_src.name
    shutil.copy2(texture_src, texture_dst)
    shutil.copy2(process_src, process_dst)

    sx_text = sx_template.read_text(encoding="utf-8", errors="replace")
    sx_dst = case_dir / "sx"
    sx_dst.write_text(update_sx_voce_robust(sx_text, voce), encoding="utf-8")

    vpsc7_new = rewrite_vpsc7_paths(vpsc7_src, texture_dst.name, sx_dst.name, process_dst.name)
    (case_dir / "vpsc7.in").write_text(vpsc7_new, encoding="utf-8")
    return case_dir


def run_evpsc_in_dir(runtime: RuntimePaths, case_dir: Path, timeout_sec: int | None = None) -> None:
    if not runtime.evpsc_exe.exists():
        raise FileNotFoundError(f"EVPSC executable not found: {runtime.evpsc_exe}")
    proc = subprocess.run(
        [str(runtime.evpsc_exe)],
        cwd=str(case_dir),
        env=ensure_evpsc_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_sec,
        check=False,
    )
    (case_dir / "STDOUT_STDERR.log").write_text(proc.stdout, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"EVPSC failed in {case_dir} with code {proc.returncode}")


def detect_axial_component(str_df) -> str:
    candidates = []
    for axis in ["11", "22", "33"]:
        e_col = f"E{axis}"
        if e_col in str_df.columns:
            candidates.append((axis, float(abs(str_df.iloc[-1][e_col]))))
    if not candidates:
        for axis in ["11", "22", "33"]:
            if f"SCAU{axis}" in str_df.columns:
                return axis
        raise ValueError("Cannot detect axial component.")
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[0][0]


def extract_simulation_outputs(case_dir: Path) -> dict[str, Any]:
    str_path = case_dir / "STR_STR.OUT"
    act_path = case_dir / "ACT_PH1.OUT"
    if not str_path.exists() or not act_path.exists():
        raise FileNotFoundError(f"Missing EVPSC outputs in {case_dir}")

    str_df = read_whitespace_table(str_path)
    act_df = read_whitespace_table(act_path)
    axial = detect_axial_component(str_df)
    e_col = f"E{axial}"
    s_col = f"SCAU{axial}"
    if s_col not in str_df.columns:
        raise ValueError(f"Missing {s_col} in STR_STR.OUT")
    if "STRAIN" not in act_df.columns:
        raise ValueError("Missing STRAIN column in ACT_PH1.OUT")

    eps = act_df["STRAIN"].to_numpy(dtype=float)
    sigma_raw = str_df[s_col].to_numpy(dtype=float)
    if len(sigma_raw) != len(eps):
        if e_col not in str_df.columns:
            raise ValueError("Row mismatch and no strain column available for interpolation fallback.")
        e_comp = np.abs(str_df[e_col].to_numpy(dtype=float))
        sigma_raw = np.interp(eps, e_comp, sigma_raw)

    final_e = float(str_df.iloc[-1][e_col]) if e_col in str_df.columns else float("nan")
    sigma = -sigma_raw if np.isfinite(final_e) and final_e < 0 else sigma_raw
    order = np.argsort(eps)
    eps = eps[order]
    sigma = sigma[order]
    mode_cols = [col for col in ["MODE1", "MODE2", "MODE3", "MODE4"] if col in act_df.columns]
    modes = act_df[mode_cols].to_numpy(dtype=float)[order] if mode_cols else None
    twfr = act_df["TWFR4"].to_numpy(dtype=float)[order] if "TWFR4" in act_df.columns else None
    return {"eps": eps, "sigma": sigma, "modes": modes, "twfr": twfr}


def load_experimental_curve(runtime: RuntimePaths, case_name: str) -> tuple[np.ndarray, np.ndarray]:
    path = runtime.exp_dir / CASE_TO_EXPERIMENT_FILE[case_name]
    if not path.exists():
        raise FileNotFoundError(f"Experimental curve not found: {path}")
    return read_numeric_two_column_file(path)


def evaluate_curve_fit(sim_eps: np.ndarray, sim_sig: np.ndarray, exp_eps: np.ndarray, exp_sig: np.ndarray, sigma_scale_eps: float = DEFAULT_SIGMA_SCALE_EPS) -> dict[str, float]:
    rmse = rmse_interp(sim_eps, sim_sig, exp_eps, exp_sig)
    r2 = r2_interp(sim_eps, sim_sig, exp_eps, exp_sig)
    scale = sigma_at(sigma_scale_eps, exp_eps, exp_sig)
    dshape = drmse_scaled(sim_eps, sim_sig, exp_eps, exp_sig, scale)
    return {"rmse": float(rmse), "r2": float(r2), "dshape": float(dshape)}

# ===== END INLINED MODULE: evpsc.py =====


# ===== BEGIN INLINED MODULE: stage1.py =====


from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import qmc



class Stage1Evaluator:
    def __init__(self, runtime: RuntimePaths, options: Stage1Options, out_dir: Path):
        self.runtime = runtime
        self.options = options
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.trials_csv = self.out_dir / "trials_stage1.csv"
        self.jsonl_path = self.out_dir / "trials_stage1.jsonl"
        self.curves_dir = self.out_dir / "curves"
        if self.options.export_curves:
            self.curves_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_x16(self, x16: list[float], trial_id: int | None = None, return_details: bool = False) -> dict[str, Any]:
        if trial_id is None:
            trial_id = next_trial_id(self.trials_csv)
        voce = VoceParams.from_vector(x16)

        case_dirs = {case: prepare_case_run_dir(self.runtime, case, voce) for case in CASES}
        fail_cases: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=max(1, min(self.options.parallel_cases, len(CASES)))) as pool:
            futures = {pool.submit(run_evpsc_in_dir, self.runtime, case_dirs[case], self.options.timeout_sec): case for case in CASES}
            for future in as_completed(futures):
                case = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    fail_cases[case] = repr(exc)

        per_case: dict[str, Any] = {}
        anchors_dict: dict[str, float] = {}
        rmse_values: list[float] = []
        dshape_values: list[float] = []

        for case in CASES:
            if case in fail_cases:
                per_case[case] = {"state": "FAIL", "rmse": float("inf"), "r2": float("nan"), "dshape": float("inf"), "f0_case": float(DEFAULT_FAIL_PENALTY)}
                continue

            sim = extract_simulation_outputs(case_dirs[case])
            exp_eps, exp_sig = load_experimental_curve(self.runtime, case)
            fit = evaluate_curve_fit(sim["eps"], sim["sigma"], exp_eps, exp_sig)

            rmse = float(fit["rmse"])
            dshape = float(fit["dshape"])
            rmse_values.append(rmse)
            dshape_values.append(dshape)
            f0_case = float(self.options.w_rmse * rmse + self.options.w_dshape * dshape)

            if sim["modes"] is None or np.asarray(sim["modes"]).size == 0:
                mode_raw = np.zeros((len(sim["eps"]), 4), dtype=float)
            else:
                mr = np.asarray(sim["modes"], dtype=float)
                mode_raw = np.zeros((mr.shape[0], 4), dtype=float)
                if mr.ndim == 2:
                    mode_raw[:, : min(4, mr.shape[1])] = mr[:, : min(4, mr.shape[1])]
            mode_raw = np.clip(mode_raw, 0.0, None)
            mode_frac = modes_to_fractions(mode_raw) if len(mode_raw) else np.zeros((0, 4), dtype=float)
            twfr_raw = sim["twfr"] if sim["twfr"] is not None else np.full(len(sim["eps"]), np.nan, dtype=float)
            proc = preprocess_modes_piecewise_raw(sim["eps"], mode_raw, twfr_raw)
            rel_anchor = relative_fractions_at_anchors_from_raw(proc["eps"], proc["mode_raw"], ANCHORS)
            twfr_anchor = sample_at_anchors(proc["eps"], proc["twfr"], ANCHORS) if np.any(np.isfinite(proc["twfr"])) else [float("nan")] * len(ANCHORS)

            for i_anchor, anchor in enumerate(ANCHORS):
                tag = anchor_tag(anchor)
                for i_mech, mech in enumerate(MECH_ORDER):
                    raw_value = float(np.interp(float(anchor), proc["eps"], proc["mode_raw"][:, i_mech])) if float(anchor) >= float(proc["eps"][0]) and float(anchor) <= float(proc["eps"][-1]) else float("nan")
                    anchors_dict[f"act_raw_{case}_{mech}_{tag}"] = raw_value
                    anchors_dict[f"act_rel_{case}_{mech}_{tag}"] = float(rel_anchor[i_anchor, i_mech]) if np.isfinite(rel_anchor[i_anchor, i_mech]) else float("nan")
                anchors_dict[f"twfr4_{case}_{tag}"] = float(twfr_anchor[i_anchor])

            per_case[case] = {
                "state": "COMPLETE",
                "rmse": rmse,
                "r2": float(fit["r2"]),
                "dshape": dshape,
                "f0_case": f0_case,
                "sim_eps": sim["eps"],
                "sim_sig": sim["sigma"],
                "exp_eps": exp_eps,
                "exp_sig": exp_sig,
            }

        if fail_cases:
            f0 = float(DEFAULT_FAIL_PENALTY)
            f1 = float(DEFAULT_FAIL_PENALTY)
            prior_valid = 0
            state = "FAIL"
            term_details = {f"term{i}": float(DEFAULT_FAIL_PENALTY) for i in range(1, 7)}
        else:
            f0_rmse_mean_raw = float(np.mean(rmse_values)) if rmse_values else float("inf")
            f0_dshape_mean_raw = float(np.mean(dshape_values)) if dshape_values else float("inf")
            f0 = float(self.options.w_rmse * f0_rmse_mean_raw + self.options.w_dshape * f0_dshape_mean_raw)
            f1, term_details = prior_penalty_v2(anchors_dict)
            prior_valid = int(f1 < DEFAULT_FAIL_PENALTY)
            state = "COMPLETE"

        row: dict[str, Any] = {
            "trial": int(trial_id),
            "state": state,
            "f0": float(f0),
            "f1": float(f1),
            "prior_valid": int(prior_valid),
            "f0_rmse_mean_raw": float(np.mean(rmse_values)) if rmse_values else float("inf"),
            "f0_dshape_mean_raw": float(np.mean(dshape_values)) if dshape_values else float("inf"),
            "f0_rmse_mean": float(self.options.w_rmse * np.mean(rmse_values)) if rmse_values else float("inf"),
            "f0_dshape_mean": float(self.options.w_dshape * np.mean(dshape_values)) if dshape_values else float("inf"),
            **VoceParams.from_vector(x16).to_flat_dict(),
            **anchors_dict,
            **term_details,
        }
        for case in CASES:
            pc = per_case.get(case, {})
            row[f"f0_rmse_{case}_raw"] = float(pc.get("rmse", float("inf")))
            row[f"f0_dshape_{case}_raw"] = float(pc.get("dshape", float("inf")))
            row[f"f0_case_{case}"] = float(pc.get("f0_case", float(DEFAULT_FAIL_PENALTY)))
            row[f"r2_{case}"] = float(pc.get("r2", float("nan")))
        append_row_fixed_atomic(self.trials_csv, row, STAGE1_SCHEMA)
        append_jsonl_row(self.jsonl_path, row, STAGE1_SCHEMA)

        if self.options.export_curves and not fail_cases:
            payload = {}
            for case, pc in per_case.items():
                payload[f"{case}/sim_eps"] = pc["sim_eps"]
                payload[f"{case}/sim_sig"] = pc["sim_sig"]
                payload[f"{case}/exp_eps"] = pc["exp_eps"]
                payload[f"{case}/exp_sig"] = pc["exp_sig"]
            np.savez_compressed(self.curves_dir / f"trial{trial_id:06d}_curves.npz", **payload)


        if return_details:
            return {"row": row, "per_case": per_case, "anchors": anchors_dict, "fail_cases": fail_cases}

        return row


def dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (float(a["f0"]) <= float(b["f0"]) and float(a["f1"]) <= float(b["f1"])) and (float(a["f0"]) < float(b["f0"]) or float(a["f1"]) < float(b["f1"]))


def fast_non_dominated_sort(pop: list[dict[str, Any]]) -> list[list[int]]:
    S = [set() for _ in pop]
    n = [0 for _ in pop]
    fronts: list[list[int]] = [[]]
    for p in range(len(pop)):
        for q in range(len(pop)):
            if p == q:
                continue
            if dominates(pop[p], pop[q]):
                S[p].add(q)
            elif dominates(pop[q], pop[p]):
                n[p] += 1
        if n[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        next_front: list[int] = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    return fronts[:-1]


def crowding_distance(front: list[dict[str, Any]]) -> np.ndarray:
    n = len(front)
    if n == 0:
        return np.zeros(0, dtype=float)
    dist = np.zeros(n, dtype=float)
    for key in ["f0", "f1"]:
        order = np.argsort([float(item[key]) for item in front])
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        values = np.asarray([float(front[idx][key]) for idx in order], dtype=float)
        denom = values[-1] - values[0]
        if denom <= 1e-12:
            continue
        for i in range(1, n - 1):
            dist[order[i]] += (values[i + 1] - values[i - 1]) / denom
    return dist


def tournament_select(pop: list[dict[str, Any]], fronts: list[list[int]], crowding: dict[int, float], rng: np.random.RandomState) -> dict[str, Any]:
    i, j = rng.randint(0, len(pop)), rng.randint(0, len(pop))
    rank = {}
    for r, front in enumerate(fronts):
        for idx in front:
            rank[idx] = r
    if rank[i] < rank[j]:
        return pop[i]
    if rank[j] < rank[i]:
        return pop[j]
    return pop[i] if crowding[i] >= crowding[j] else pop[j]


def blend_crossover(u1: np.ndarray, u2: np.ndarray, rng: np.random.RandomState, alpha: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    low = np.minimum(u1, u2)
    high = np.maximum(u1, u2)
    delta = high - low
    child1 = rng.uniform(low - alpha * delta, high + alpha * delta)
    child2 = rng.uniform(low - alpha * delta, high + alpha * delta)
    return mirror01(child1), mirror01(child2)


def gaussian_mutation(u: np.ndarray, rng: np.random.RandomState, sigma: float = 0.08, p: float = 0.20) -> np.ndarray:
    out = np.asarray(u, dtype=float).copy()
    mask = rng.rand(out.size) < float(p)
    out[mask] += rng.normal(0.0, sigma, size=int(mask.sum()))
    return mirror01(out)


def run_stage1_biobjective_search(
    evaluator: Stage1Evaluator,
    *,
    n_total: int,
    n_sobol: int,
    population_size: int,
    seed: int,
) -> pd.DataFrame:
    specs = build_var_specs(theta_scale="log")
    rng = np.random.RandomState(int(seed))
    sobol = qmc.Sobol(d=16, scramble=True, seed=int(seed))
    n_sobol_eff = int(n_sobol)
    u_sobol = sobol.random_base2(m=int(np.ceil(np.log2(max(1, n_sobol_eff)))))[:n_sobol_eff]

    population: list[dict[str, Any]] = []
    for u in u_sobol:
        row = evaluator.evaluate_x16(unit_to_x16(u, specs))
        population.append({**row, "u": np.asarray(u, dtype=float)})
        if len(population) >= n_total:
            break

    if len(population) >= n_total:
        return pd.DataFrame([{k: v for k, v in item.items() if k != "u"} for item in population])

    while len(population) < n_total:
        fronts = fast_non_dominated_sort(population)
        crowding = {}
        for front in fronts:
            distances = crowding_distance([population[idx] for idx in front])
            for idx, distance in zip(front, distances):
                crowding[idx] = float(distance)
        mating_pool = [tournament_select(population, fronts, crowding, rng) for _ in range(population_size)]
        offspring_rows: list[dict[str, Any]] = []
        for i in range(0, len(mating_pool), 2):
            parent1 = mating_pool[i]
            parent2 = mating_pool[(i + 1) % len(mating_pool)]
            c1_u, c2_u = blend_crossover(parent1["u"], parent2["u"], rng)
            c1_u = gaussian_mutation(c1_u, rng)
            c2_u = gaussian_mutation(c2_u, rng)
            for child_u in [c1_u, c2_u]:
                if len(population) + len(offspring_rows) >= n_total:
                    break
                row = evaluator.evaluate_x16(unit_to_x16(child_u, specs))
                offspring_rows.append({**row, "u": child_u})
            if len(population) + len(offspring_rows) >= n_total:
                break

        combined = population + offspring_rows
        fronts = fast_non_dominated_sort(combined)
        new_population: list[dict[str, Any]] = []
        for front in fronts:
            if len(new_population) + len(front) <= population_size:
                new_population.extend(combined[idx] for idx in front)
            else:
                front_items = [combined[idx] for idx in front]
                distances = crowding_distance(front_items)
                order = np.argsort(-distances)
                slots = population_size - len(new_population)
                new_population.extend(front_items[idx] for idx in order[:slots])
                break
        population = new_population

    return pd.DataFrame([{k: v for k, v in item.items() if k != "u"} for item in population])

# ===== END INLINED MODULE: stage1.py =====


# ===== BEGIN INLINED MODULE: polish.py =====


import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


try:
    import umap  # type: ignore
    HAS_UMAP = True
except Exception:
    umap = None
    HAS_UMAP = False


@dataclass
class DistLockBundle:
    k_star: int
    prior_feasible: float
    top_frac: float
    scaler: StandardScaler
    mu_z: dict[int, np.ndarray]
    var_z: dict[int, np.ndarray]
    d2_gate: dict[int, float]
    elite_df: pd.DataFrame
    umap_model: Any
    umap_xy: np.ndarray | None


@dataclass
class EvalResult:
    u: np.ndarray
    x16: list[float]
    f0: float
    f1: float
    d2: float
    d2n: float
    J: float
    ok: bool
    reason: str
    meta: dict[str, Any]


@dataclass
class CMAES:
    x0_u: np.ndarray
    sigma0: float
    seed: int

    def __post_init__(self):
        self.rng = np.random.RandomState(int(self.seed))
        self.n = int(self.x0_u.size)
        self.lambda_ = 4 + int(3 * np.log(self.n))
        self.mu = self.lambda_ // 2
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = weights / np.sum(weights)
        self.m = self.x0_u.astype(float).copy()
        self.sigma = float(self.sigma0)
        self.C = np.eye(self.n)
        self.pc = np.zeros(self.n)
        self.ps = np.zeros(self.n)
        self.cs = 0.3
        self.cc = 0.2
        self.c1 = 0.2
        self.cmu = 0.2
        self.damps = 1.2
        self.chiN = math.sqrt(self.n) * (1 - 1 / (4 * self.n) + 1 / (21 * self.n**2))

    def ask(self) -> np.ndarray:
        A = np.linalg.cholesky(self.C)
        z = self.rng.randn(self.lambda_, self.n)
        return self.m + self.sigma * z.dot(A.T)

    def tell(self, X: np.ndarray, f: np.ndarray) -> None:
        idx = np.argsort(f)
        Xs = X[idx]
        m_old = self.m.copy()
        self.m = np.sum(Xs[: self.mu] * self.w.reshape(-1, 1), axis=0)
        y = (self.m - m_old) / self.sigma
        C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(self.C)).T
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mu) * (C_inv_sqrt @ y)
        hsig = float(np.linalg.norm(self.ps) / self.chiN) < (1.4 + 2 / (self.n + 1))
        self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(self.cc * (2 - self.cc) * self.mu) * y
        artmp = (Xs[: self.mu] - m_old) / self.sigma
        self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * np.outer(self.pc, self.pc)
        for i in range(self.mu):
            self.C += self.cmu * self.w[i] * np.outer(artmp[i], artmp[i])
        self.sigma *= math.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))


def read_kv_csv(path: Path) -> dict[str, str]:
    df = pd.read_csv(path)
    if "key" not in df.columns or "value" not in df.columns:
        return {}
    return {str(r["key"]).strip(): str(r["value"]).strip() for _, r in df.iterrows()}


def d2_mahalanobis(z: np.ndarray, mu: np.ndarray, var: np.ndarray) -> float:
    return float(np.sum(((z - mu) ** 2) / (var + 1e-12)))


def load_stage2_bundle(stage2_dir: Path, d2_gate_q: float, random_state: int = 0) -> DistLockBundle:
    kv = read_kv_csv(stage2_dir / "run_summary.csv")
    k_star = int(float(kv.get("k_star", "4")))
    prior_feasible = float(kv.get("prior_feasible", "0.05"))
    top_frac = float(kv.get("top_frac", "0.10"))
    elite = safe_read_csv(stage2_dir / f"subset_points_full_k{k_star}.csv")
    elite["trial"] = pd.to_numeric(elite["trial"], errors="coerce").astype(int)
    elite["cluster"] = pd.to_numeric(elite["cluster"], errors="coerce").astype(int)
    X = build_clr_features_from_df(elite, eps=1e-12)
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)

    mu_z: dict[int, np.ndarray] = {}
    var_z: dict[int, np.ndarray] = {}
    d2_gate: dict[int, float] = {}
    for cl in sorted(elite["cluster"].unique().astype(int).tolist()):
        idx = elite.index[elite["cluster"] == int(cl)].to_numpy(int)
        if len(idx) < 6:
            continue
        Zk = Z[idx]
        mu = np.mean(Zk, axis=0)
        var = np.var(Zk, axis=0, ddof=1)
        var[var < 1e-8] = 1e-8
        mu_z[int(cl)] = mu.astype(float)
        var_z[int(cl)] = var.astype(float)
        d2 = np.sum(((Zk - mu) ** 2) / var, axis=1)
        d2_gate[int(cl)] = float(np.quantile(d2[np.isfinite(d2)], float(d2_gate_q)))

    umap_model = None
    umap_xy = None
    if HAS_UMAP:
        try:
            umap_model = umap.UMAP(
                n_components=2,
                n_neighbors=int(float(kv.get("umap_n_neighbors", "40"))),
                min_dist=float(kv.get("umap_min_dist", "0.05")),
                metric=str(kv.get("umap_metric", "euclidean")),
                random_state=int(random_state),
            )
            umap_xy = umap_model.fit_transform(Z)
            elite["u1"] = umap_xy[:, 0]
            elite["u2"] = umap_xy[:, 1]
        except Exception:
            umap_model = None
            umap_xy = None

    return DistLockBundle(k_star, prior_feasible, top_frac, scaler, mu_z, var_z, d2_gate, elite, umap_model, umap_xy)


def objective_distance_lock(f0: float, f1: float, d2: float, d2_gate: float, prior_feasible: float, lam_c: float) -> tuple[float, bool, str, float]:
    if not (np.isfinite(f0) and np.isfinite(f1) and np.isfinite(d2)):
        return float(DEFAULT_FAIL_PENALTY), False, "nonfinite", float("nan")
    if f0 >= 9.9e5 or f1 >= 9.9e5:
        return float(DEFAULT_FAIL_PENALTY), False, "evpsc_fail", float("nan")
    if not np.isfinite(d2_gate) or d2_gate <= 0:
        return float(DEFAULT_FAIL_PENALTY), False, "bad_gate", float("nan")
    d2n = float(d2) / float(d2_gate + 1e-12)
    dterm = float(lam_c) * float(d2n)
    if f1 > float(prior_feasible):
        return float(DEFAULT_FAIL_PENALTY) + 1.0e3 * float(f1 - prior_feasible) + 1.0e-3 * float(f0) + dterm, False, "prior", d2n
    if d2 > float(d2_gate):
        return float(DEFAULT_FAIL_PENALTY) + 1.0e2 * float(d2n - 1.0) + 1.0e-3 * float(f0) + dterm, False, "dist", d2n
    return float(f0) + dterm, True, "ok", d2n


def trial_to_x16_from_trials(trials_df: pd.DataFrame, trial_id: int) -> list[float]:
    row = trials_df.loc[trials_df["trial"] == int(trial_id)]
    if len(row) == 0:
        raise ValueError(f"Trial not found: {trial_id}")
    r = row.iloc[0]
    out = []
    for mech in MECH_ORDER:
        for key in ["tau0", "tau1", "thet0", "thet1"]:
            out.append(float(r[f"{mech}_{key}"]))
    return out


def evaluate_polish_candidate(evaluator: Stage1Evaluator, x16: list[float]) -> tuple[float, float, dict[str, float], dict[str, Any]]:
    row = evaluator.evaluate_x16(x16)
    anchors = {key: float(row[key]) for key in row if key.startswith("act_rel_")}
    f0 = float(row["f0"])
    f1 = float(row["f1"])
    term_details = prior_penalty_v2(anchors)[1]
    return f0, f1, anchors, {"rmse_mean": float(row["f0_rmse_mean_raw"]), "dshape_mean": float(row["f0_dshape_mean_raw"]), **term_details}


def plot_trajectories(df_log: pd.DataFrame, out_dir: Path, label: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df_log.sort_values(["gen", "eval"]).reset_index(drop=True)
    x = df["eval"].to_numpy(int)
    if "rmse_mean" in df.columns and "d2n_target" in df.columns:
        rmse = np.minimum.accumulate(np.where(np.isfinite(df["rmse_mean"].to_numpy(float)), df["rmse_mean"].to_numpy(float), np.inf))
        d2n = df["d2n_target"].to_numpy(float)
        fig, ax1 = plt.subplots(figsize=(6.2, 3.6))
        ax1.plot(x, rmse)
        ax1.set_xlabel("Evaluation index")
        ax1.set_ylabel("RMSE (best-so-far)")
        ax2 = ax1.twinx()
        ax2.plot(x, d2n)
        ax2.set_ylabel("d2 / d2_gate")
        ax1.set_title(label)
        plt.tight_layout()
        plt.savefig(out_dir / f"fig_traj_rmse_d2n_{label}.png", dpi=220, bbox_inches="tight")
        plt.close()
    if "f1" in df.columns:
        fig, ax = plt.subplots(figsize=(6.2, 3.2))
        ax.plot(x, df["f1"].to_numpy(float))
        ax.set_xlabel("Evaluation index")
        ax.set_ylabel("f1")
        ax.set_title(label)
        plt.tight_layout()
        plt.savefig(out_dir / f"fig_traj_f1_{label}.png", dpi=220, bbox_inches="tight")
        plt.close()


def export_curves(evaluator: Stage1Evaluator, x_start: list[float], x_best: list[float], curves_dir: Path, plots_dir: Path, label: str) -> None:
    curves_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    start_detail = evaluator.evaluate_x16(x_start, return_details=True)
    best_detail = evaluator.evaluate_x16(x_best, return_details=True)
    for case in CASES:
        pc0 = start_detail["per_case"].get(case, {})
        pc1 = best_detail["per_case"].get(case, {})
        if not pc0 or not pc1 or pc0.get("state") != "COMPLETE" or pc1.get("state") != "COMPLETE":
            continue
        exp_eps = pc0["exp_eps"]
        exp_sig = pc0["exp_sig"]
        sim0_eps = pc0["sim_eps"]
        sim0_sig = pc0["sim_sig"]
        sim1_eps = pc1["sim_eps"]
        sim1_sig = pc1["sim_sig"]
        rows = []
        for eps, sig in zip(exp_eps, exp_sig):
            rows.append({"curve": "exp", "eps": float(eps), "sigma": float(sig)})
        for eps, sig in zip(sim0_eps, sim0_sig):
            rows.append({"curve": "start", "eps": float(eps), "sigma": float(sig)})
        for eps, sig in zip(sim1_eps, sim1_sig):
            rows.append({"curve": "polished", "eps": float(eps), "sigma": float(sig)})
        pd.DataFrame(rows).to_csv(curves_dir / f"curves_{case}_{label}.csv", index=False, encoding="utf-8")
        fig, ax = plt.subplots(figsize=(5.3, 3.8))
        ax.plot(exp_eps, exp_sig, label="exp")
        ax.plot(sim0_eps, sim0_sig, label="start")
        ax.plot(sim1_eps, sim1_sig, label="polished")
        ax.set_xlabel("Strain")
        ax.set_ylabel("Stress")
        ax.set_title(f"{case} | {label}")
        ax.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(plots_dir / f"fig_curves_{case}_{label}.png", dpi=220, bbox_inches="tight")
        plt.close()


def run_polish_one(bundle: DistLockBundle, specs: list[VarSpec], target_cluster: int, rep_type: str, x0: list[float], evaluator: Stage1Evaluator, out_dir: Path, options: PolishingOptions) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    plot_dir = out_dir / "plots"
    model_dir = out_dir / "models"
    curves_dir = out_dir / "curves"
    for d in [log_dir, plot_dir, model_dir, curves_dir]:
        d.mkdir(parents=True, exist_ok=True)

    mu = bundle.mu_z.get(int(target_cluster))
    var = bundle.var_z.get(int(target_cluster))
    d2_gate = float(bundle.d2_gate.get(int(target_cluster), float("nan")))
    if mu is None or var is None or not np.isfinite(d2_gate):
        raise RuntimeError(f"Missing cluster statistics for cluster={target_cluster}")

    cma = CMAES(x0_u=mirror01(x16_to_unit(x0, specs)), sigma0=float(options.sigma0), seed=int(options.seed) + 100 * int(target_cluster))
    rows_log: list[dict[str, Any]] = []
    cache: dict[tuple[float, ...], EvalResult] = {}
    best: EvalResult | None = None
    bestJ = float("inf")
    stall = 0
    eval_i = 0
    gen = 0

    def _eval_u(u: np.ndarray, injected: int, i_in_gen: int) -> EvalResult:
        nonlocal eval_i, best, bestJ, stall, gen
        key = tuple(np.round(u.astype(float), 8).tolist())
        if key in cache:
            res = cache[key]
        else:
            x16 = unit_to_x16(u, specs)
            f0, f1, anchors, meta = evaluate_polish_candidate(evaluator, x16)
            feat = build_clr_features_from_anchor_dict(anchors, eps=1e-12)
            d2 = float("inf") if not np.isfinite(feat).all() else d2_mahalanobis(bundle.scaler.transform(feat.reshape(1, -1)).reshape(-1), mu, var)
            J, ok, reason, d2n = objective_distance_lock(f0, f1, d2, d2_gate, bundle.prior_feasible, options.lam_c)
            res = EvalResult(u.copy(), x16, float(f0), float(f1), float(d2), float(d2n), float(J), bool(ok), str(reason), meta)
            cache[key] = res
        rows_log.append({
            "eval": int(eval_i),
            "gen": int(gen),
            "i_in_gen": int(i_in_gen),
            "injected": int(injected),
            "sigma": float(cma.sigma),
            "cluster_target": int(target_cluster),
            "rep_type": str(rep_type),
            "f0": float(res.f0),
            "rmse_mean": float(res.meta.get("rmse_mean", float("nan"))),
            "dshape_mean": float(res.meta.get("dshape_mean", float("nan"))),
            "f1": float(res.f1),
            "d2_target": float(res.d2),
            "d2n_target": float(res.d2n),
            "d2_gate": float(d2_gate),
            "J": float(res.J),
            "ok": int(res.ok),
            "reason": str(res.reason),
        })
        if res.ok:
            if res.J < bestJ:
                bestJ = float(res.J)
                best = res
                stall = 0
            else:
                stall += 1
        eval_i += 1
        return res

    _eval_u(mirror01(cma.m.copy()), 1, -1)
    while eval_i < int(options.nmax) and stall < int(options.stall_evals):
        gen += 1
        injections = [mirror01(cma.m.copy())]
        if best is not None:
            injections.append(best.u.copy())
        injections.append(mirror01(x16_to_unit(x0, specs)))
        pool_u: list[np.ndarray] = []
        pool_J: list[float] = []
        i_in_gen = 0
        for u_inj in injections:
            if eval_i >= int(options.nmax) or stall >= int(options.stall_evals):
                break
            res = _eval_u(u_inj, 1, i_in_gen)
            i_in_gen += 1
            pool_u.append(res.u)
            pool_J.append(res.J)
        for ask in cma.ask():
            if eval_i >= int(options.nmax) or stall >= int(options.stall_evals):
                break
            res = _eval_u(mirror01(ask), 0, i_in_gen)
            i_in_gen += 1
            pool_u.append(res.u)
            pool_J.append(res.J)
        if len(pool_J) >= max(2, cma.mu):
            cma.tell(np.vstack(pool_u).astype(float), np.asarray(pool_J, dtype=float))

    df_log = pd.DataFrame(rows_log)
    label = f"cluster{int(target_cluster)}_{rep_type}"
    df_log.to_csv(log_dir / f"polish_log_{label}.csv", index=False, encoding="utf-8")
    plot_trajectories(df_log, plot_dir, label)
    pd.DataFrame({
        "cluster": [int(target_cluster)],
        "d2_gate": [float(d2_gate)],
        "n_elite": [int(np.sum(bundle.elite_df["cluster"].to_numpy(int) == int(target_cluster)))],
    }).to_csv(model_dir / f"cluster_stats_distlock_k{bundle.k_star}.csv", index=False, encoding="utf-8")

    if best is None:
        return {"cluster": int(target_cluster), "rep_type": rep_type, "status": "NO_FEASIBLE_FOUND", "n_eval": int(eval_i), "bestJ": float("inf")}

    export_curves(evaluator, x0, best.x16, curves_dir, plot_dir, label)
    return {
        "cluster": int(target_cluster),
        "rep_type": str(rep_type),
        "status": "OK",
        "n_eval": int(eval_i),
        "bestJ": float(bestJ),
        "best_f0": float(best.f0),
        "best_f1": float(best.f1),
        "best_d2": float(best.d2),
        "best_d2n": float(best.d2n),
        **{f"best_x{i:02d}": float(best.x16[i]) for i in range(16)},
    }


def run_stage2_polish(stage2_dir: Path, trials_csv: Path, runtime: RuntimePaths, out_dir: Path, options: PolishingOptions) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_stage2_bundle(stage2_dir, d2_gate_q=float(options.d2n_gate_q), random_state=int(options.random_state))
    reps_csv = stage2_dir / f"representatives_k{bundle.k_star}.csv"
    reps_df = safe_read_csv(reps_csv)
    if "rep_type" not in reps_df.columns and "rep" in reps_df.columns:
        reps_df = reps_df.rename(columns={"rep": "rep_type"})
    reps_df = reps_df[reps_df["rep_type"].isin(list(options.rep_types))].copy()
    if len(reps_df) == 0:
        raise RuntimeError("No representatives matched the requested rep_types.")

    trials_df = safe_read_csv(trials_csv)
    trials_df["trial"] = pd.to_numeric(trials_df["trial"], errors="coerce")
    trials_df = trials_df[trials_df["trial"].notna()].copy()
    trials_df["trial"] = trials_df["trial"].astype(int)

    stage1_options = Stage1Options(timeout_sec=options.timeout_sec, parallel_cases=options.parallel_cases, w_rmse=options.w_rmse, w_dshape=options.w_dshape, export_curves=False)
    evaluator = Stage1Evaluator(runtime, stage1_options, out_dir / "_stage1_eval_cache")
    specs = build_var_specs(theta_scale="log")
    summary_rows = []

    for _, rep_row in reps_df.iterrows():
        cl = int(rep_row["cluster"])
        rep_type = str(rep_row["rep_type"])
        x0 = trial_to_x16_from_trials(trials_df, int(rep_row["trial"]))
        summary = run_polish_one(bundle, specs, cl, rep_type, x0, evaluator, out_dir, options)
        summary.update({
            "trial_seed": int(rep_row["trial"]),
            "stage2_dir": str(stage2_dir),
            "k_star": int(bundle.k_star),
            "prior_feasible": float(bundle.prior_feasible),
            "top_frac": float(bundle.top_frac),
            "d2_gate_q": float(options.d2n_gate_q),
        })
        summary_rows.append(summary)

    pd.DataFrame(summary_rows).to_csv(out_dir / "polish_summary.csv", index=False, encoding="utf-8")
    (out_dir / "run_meta.json").write_text(json.dumps({
        "stage2_dir": str(stage2_dir),
        "trials_csv": str(trials_csv),
        "k_star": int(bundle.k_star),
        "prior_feasible": float(bundle.prior_feasible),
        "top_frac": float(bundle.top_frac),
        "rep_types": list(options.rep_types),
        "d2_gate_q": float(options.d2n_gate_q),
        "nmax": int(options.nmax),
        "stall": int(options.stall_evals),
        "sigma0": float(options.sigma0),
        "timeout_sec": int(options.timeout_sec),
        "parallel_cases": int(options.parallel_cases),
        "lam_c": float(options.lam_c),
        "random_state": int(options.random_state),
    }, indent=2), encoding="utf-8")
    return out_dir

# ===== END INLINED MODULE: polish.py =====


# ===== BEGIN ENTRYPOINT: run_stage2_polish.py =====


import argparse
from pathlib import Path



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2-dir", type=Path, required=True)
    ap.add_argument("--trials-csv", type=Path, required=True)
    ap.add_argument("--evpsc-root", type=Path, required=True)
    ap.add_argument("--evpsc-exe", type=Path, required=True)
    ap.add_argument("--exp-dir", type=Path, required=True)
    ap.add_argument("--template-dir", type=Path, required=True)
    ap.add_argument("--scratch-dir", type=Path, default=Path("./scratch_stage2_polish"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--rep-types", type=str, default="best_balanced")
    ap.add_argument("--d2n-gate-q", type=float, default=0.95)
    ap.add_argument("--nmax", type=int, default=200)
    ap.add_argument("--stall", type=int, default=50)
    ap.add_argument("--sigma0", type=float, default=0.08)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--parallel-cases", type=int, default=4)
    ap.add_argument("--w-rmse", type=float, default=1.0)
    ap.add_argument("--w-dshape", type=float, default=1.0)
    ap.add_argument("--lam-c", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--random-state", type=int, default=0)
    args = ap.parse_args()

    runtime = RuntimePaths(args.evpsc_root, args.evpsc_exe, args.exp_dir, args.template_dir, args.scratch_dir)
    options = PolishingOptions(
        timeout_sec=int(args.timeout),
        parallel_cases=int(args.parallel_cases),
        w_rmse=float(args.w_rmse),
        w_dshape=float(args.w_dshape),
        lam_c=float(args.lam_c),
        nmax=int(args.nmax),
        stall_evals=int(args.stall),
        sigma0=float(args.sigma0),
        d2n_gate_q=float(args.d2n_gate_q),
        seed=int(args.seed),
        random_state=int(args.random_state),
        rep_types=tuple(s.strip() for s in str(args.rep_types).split(",") if s.strip()),
    )
    run_stage2_polish(args.stage2_dir, args.trials_csv, runtime, args.out_dir, options)
    print(f"Done. Outputs written to: {args.out_dir}")


if __name__ == "__main__":
    main()

# ===== END ENTRYPOINT: run_stage2_polish.py =====
