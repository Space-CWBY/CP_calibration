# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Standalone Stage-2 clustering driver (CLR-GMM; UMAP for visualization only).

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


# ===== BEGIN INLINED MODULE: clustering.py =====


import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler



def gmm_fit_safe(X: np.ndarray, k: int, random_state: int, reg_list=(1e-6, 1e-5, 1e-4, 1e-3, 1e-2), n_init: int = 10, max_iter: int = 900) -> tuple[GaussianMixture, float]:
    last_error = None
    for reg in reg_list:
        try:
            gmm = GaussianMixture(
                n_components=int(k),
                covariance_type="diag",
                random_state=int(random_state),
                init_params="random_from_data",
                n_init=int(n_init),
                reg_covar=float(reg),
                max_iter=int(max_iter),
            )
            gmm.fit(X)
            return gmm, float(reg)
        except ValueError as exc:
            last_error = exc
            continue
    raise last_error if last_error is not None else RuntimeError("GMM fitting failed.")


def smooth_1d(values: np.ndarray, win: int = 3) -> np.ndarray:
    if win <= 1 or len(values) < 3:
        return values.copy()
    pad = win // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, np.ones(win) / win, mode="valid")


def pick_elbow_k_from_bic(ms_ok: pd.DataFrame, smooth_window: int = 3) -> int:
    ks = ms_ok.sort_values("k")["k"].to_numpy(int)
    bic = ms_ok.sort_values("k")["bic"].to_numpy(float)
    if len(ks) < 3:
        return int(ms_ok.loc[ms_ok["bic"].idxmin(), "k"])
    bic_s = smooth_1d(bic, win=smooth_window)
    d2 = bic_s[2:] - 2 * bic_s[1:-1] + bic_s[:-2]
    if len(d2) == 0 or not np.any(np.isfinite(d2)):
        return int(ms_ok.loc[ms_ok["bic"].idxmin(), "k"])
    return int(ks[int(np.argmax(d2)) + 1])


def choose_k_elbow_plus_stability(ms: pd.DataFrame, models: dict[int, GaussianMixture], k_plus: int = 2, smooth_window: int = 3) -> tuple[int, int, list[int]]:
    ms_ok = ms[np.isfinite(ms["bic"].to_numpy())].copy()
    if len(ms_ok) == 0:
        raise RuntimeError("No valid GMM models were fitted.")
    k_elbow = pick_elbow_k_from_bic(ms_ok, smooth_window)
    ks_ok = set(int(k) for k in ms_ok["k"].tolist())
    candidates = [k for k in range(int(k_elbow), int(k_elbow) + int(k_plus) + 1) if k in ks_ok and k in models]
    if not candidates:
        return int(ms_ok.loc[ms_ok["bic"].idxmin(), "k"]), k_elbow, []
    sub = ms_ok[ms_ok["k"].isin(candidates)].copy()
    if not np.any(np.isfinite(sub["stability_mean"].to_numpy())):
        return int(sub.loc[sub["bic"].idxmin(), "k"]), k_elbow, candidates
    sub = sub[np.isfinite(sub["stability_mean"].to_numpy())].sort_values(["stability_mean", "bic", "k"], ascending=[False, True, True])
    return int(sub.iloc[0]["k"]), k_elbow, candidates


def map_labels_hungarian(ref: np.ndarray, pred: np.ndarray) -> np.ndarray:
    ref_ids = np.unique(ref)
    pred_ids = np.unique(pred)
    matrix = np.zeros((len(ref_ids), len(pred_ids)), dtype=int)
    for i, ref_label in enumerate(ref_ids):
        for j, pred_label in enumerate(pred_ids):
            matrix[i, j] = int(np.sum((ref == ref_label) & (pred == pred_label)))
    row_ind, col_ind = linear_sum_assignment(-matrix)
    mapping = {int(pred_ids[c]): int(ref_ids[r]) for r, c in zip(row_ind, col_ind)}
    return np.asarray([mapping.get(int(x), int(x)) for x in pred], dtype=int)


def bootstrap_stability_ari(Z: np.ndarray, k: int, random_state: int, n_boot: int = 30, subsample_frac: float = 0.85) -> tuple[float, float, np.ndarray, int, np.ndarray]:
    n = Z.shape[0]
    n_sub = max(5, int(round(float(subsample_frac) * n)))
    rng = np.random.default_rng(int(random_state))
    ref_gmm, _ = gmm_fit_safe(Z, int(k), random_state=int(random_state))
    ref_labels = ref_gmm.predict(Z).astype(int)
    ari_values = []
    hits = np.zeros(n, dtype=float)
    n_eff = 0
    for b in range(int(n_boot)):
        idx = rng.choice(n, size=n_sub, replace=False)
        try:
            gmm_b, _ = gmm_fit_safe(Z[idx], int(k), random_state=int(random_state) + 1000 * int(k) + int(b))
            pred = gmm_b.predict(Z).astype(int)
            ari_values.append(float(adjusted_rand_score(ref_labels, pred)))
            hits += (map_labels_hungarian(ref_labels, pred) == ref_labels).astype(float)
            n_eff += 1
        except Exception:
            continue
    if n_eff == 0:
        return float("nan"), float("nan"), np.full(n, np.nan), 0, ref_labels
    ari = np.asarray(ari_values, dtype=float)
    return float(np.mean(ari)), float(np.std(ari)), hits / float(n_eff), int(n_eff), ref_labels


def pick_representatives(sel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cl, group in sel.groupby("cluster"):
        g = group.copy()
        g["rank_f0"] = g["f0"].rank(method="average", ascending=True)
        g["rank_f1"] = g["f1"].rank(method="average", ascending=True)
        g["rank_sum"] = g["rank_f0"] + g["rank_f1"]
        best_balanced = g.sort_values(["rank_sum", "f0", "f1"]).iloc[0]
        rows.append(
            {
                "cluster": int(cl),
                "rep_type": "best_balanced",
                "trial": int(best_balanced["trial"]),
                "f0": float(best_balanced["f0"]),
                "f1": float(best_balanced["f1"]),
                "rmse_mean": float(best_balanced["rmse_mean"]),
                "u1": float(best_balanced["u1"]),
                "u2": float(best_balanced["u2"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["cluster", "rep_type"]).reset_index(drop=True)


def export_cluster_means_anchor_long(sel: pd.DataFrame, out_csv: Path) -> None:
    rows = []
    clusters = sorted(sel["cluster"].unique())
    for case in CASES:
        for anchor in ANCHORS:
            tag = f"e{float(anchor):.2f}".replace(".", "p")
            cols = [f"act_rel_{case}_{mech}_{tag}" for mech in MECH_ORDER]
            group_mean = sel.groupby("cluster")[cols].mean().reindex(clusters)
            for cl in clusters:
                for j, mech in enumerate(MECH_ORDER):
                    rows.append({
                        "cluster": int(cl),
                        "case": case,
                        "anchor": float(anchor),
                        "mech": mech,
                        "mech_label": MECH_LABELS[j],
                        "mean_act_rel": float(group_mean.loc[cl, cols[j]]),
                    })
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")


def export_cluster_sizes(sel: pd.DataFrame, out_csv: Path) -> None:
    counts = sel["cluster"].value_counts().sort_index()
    pd.DataFrame({"cluster": counts.index.astype(int), "count": counts.values.astype(int)}).to_csv(out_csv, index=False, encoding="utf-8")


def export_gmm_params(gmm: GaussianMixture, out_csv: Path) -> None:
    rows = []
    for i in range(gmm.means_.shape[0]):
        row = {"component": int(i), "weight": float(gmm.weights_[i])}
        for j in range(gmm.means_.shape[1]):
            row[f"mean_{j:02d}"] = float(gmm.means_[i, j])
            row[f"var_{j:02d}"] = float(gmm.covariances_[i, j])
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")


def run_stage2_clustering(input_csv: Path, out_dir: Path, options: ClusteringOptions) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = safe_read_csv(input_csv)
    if "trial" not in df.columns:
        raise RuntimeError("Missing required column: trial")
    df = df[df["trial"].notna()].copy()
    df["trial"] = pd.to_numeric(df["trial"], errors="coerce")
    df = df[df["trial"].notna()].copy()
    df["trial"] = df["trial"].astype(int)
    df = df.drop_duplicates(subset=["trial"], keep="last")

    for col in ["state", "f0", "f1", "prior_valid"]:
        if col not in df.columns:
            raise RuntimeError(f"Missing required column: {col}")
    missing = [col for col in required_actrel_columns() if col not in df.columns]
    if missing:
        raise RuntimeError(f"Missing act_rel columns, first few: {missing[:8]}")

    df = df[df["state"] == "COMPLETE"].copy()
    df["f0"] = pd.to_numeric(df["f0"], errors="coerce")
    df["f1"] = pd.to_numeric(df["f1"], errors="coerce")
    df["prior_valid"] = pd.to_numeric(df["prior_valid"], errors="coerce")
    df = df[np.isfinite(df["f0"]) & np.isfinite(df["f1"])].copy()
    df = df[(df["f0"] < 1e6) & (df["f1"] < 1e6)].copy()
    df = df[df["prior_valid"] >= 0.5].copy()
    n_complete = int(len(df))

    if options.use_prior_feasible:
        df = df[df["f1"] <= float(options.prior_feasible)].copy()
    n_after_prior = int(len(df))

    df["rmse_mean"] = compute_rmse_mean(df)
    df = df[np.isfinite(df["rmse_mean"])].copy()
    threshold = float(df["rmse_mean"].quantile(float(options.top_frac)))
    sel = df[df["rmse_mean"] <= threshold].copy()
    if len(sel) < 5:
        raise RuntimeError("Selected subset is too small. Relax prior_feasible or increase top_frac.")
    n_selected = int(len(sel))

    X_clr = build_clr_features_from_df(sel, eps=1e-12)
    ok = np.isfinite(X_clr).all(axis=1)
    sel = sel.loc[ok].copy()
    X_clr = X_clr[ok]
    Z = StandardScaler().fit_transform(X_clr)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=int(options.n_neighbors),
        min_dist=float(options.min_dist),
        metric=str(options.metric),
        random_state=int(options.random_state),
        n_jobs=1,
    )
    U = reducer.fit_transform(Z)
    sel["u1"] = U[:, 0]
    sel["u2"] = U[:, 1]

    model_rows = []
    models: dict[int, GaussianMixture] = {}
    for k in range(int(options.k_min), int(options.k_max) + 1):
        try:
            gmm, reg = gmm_fit_safe(Z, k, random_state=int(options.random_state))
            labels = gmm.predict(Z).astype(int)
            bic = float(gmm.bic(Z))
            sil = float(silhouette_score(Z, labels)) if len(np.unique(labels)) > 1 else float("nan")
            model_rows.append({"k": int(k), "bic": bic, "silhouette": sil, "reg_covar_used": reg, "status": "OK"})
            models[int(k)] = gmm
        except Exception as exc:
            model_rows.append({"k": int(k), "bic": np.inf, "silhouette": np.nan, "reg_covar_used": np.nan, "status": f"FAIL:{type(exc).__name__}"})
    ms = pd.DataFrame(model_rows).sort_values("k").reset_index(drop=True)
    ms["stability_mean"] = np.nan
    ms["stability_std"] = np.nan
    ms["stability_n_eff"] = 0

    ok_ms = ms[np.isfinite(ms["bic"].to_numpy())].copy()
    if len(ok_ms) == 0:
        raise RuntimeError("All GMM fits failed.")
    k_elbow = pick_elbow_k_from_bic(ok_ms, smooth_window=int(options.elbow_smooth))
    ks_ok = set(int(k) for k in ok_ms["k"].tolist())
    candidates = [k for k in range(int(k_elbow), int(k_elbow) + int(options.elbow_plus) + 1) if k in ks_ok and k in models]

    for k in range(int(options.k_min), int(options.k_max) + 1):
        if k not in ks_ok or k not in models:
            continue
        mean_ari, std_ari, _, n_eff, _ = bootstrap_stability_ari(Z, k, random_state=int(options.random_state) + 17 * int(k), n_boot=int(options.n_boot), subsample_frac=float(options.subsample_frac))
        ms.loc[ms["k"] == int(k), "stability_mean"] = float(mean_ari)
        ms.loc[ms["k"] == int(k), "stability_std"] = float(std_ari)
        ms.loc[ms["k"] == int(k), "stability_n_eff"] = int(n_eff)

    k_star, k_elbow, candidates = choose_k_elbow_plus_stability(ms, models, k_plus=int(options.elbow_plus), smooth_window=int(options.elbow_smooth))
    gmm_star = models[int(k_star)]
    sel["cluster"] = gmm_star.predict(Z).astype(int)

    reps = pick_representatives(sel)
    reps.to_csv(out_dir / f"representatives_k{k_star}.csv", index=False, encoding="utf-8")
    core_cols = ["trial", "cluster", "rmse_mean", "f0", "f1", "u1", "u2"]
    full_cols = core_cols + required_actrel_columns()
    sel[core_cols].to_csv(out_dir / f"subset_points_core_k{k_star}.csv", index=False, encoding="utf-8")
    sel[full_cols].to_csv(out_dir / f"subset_points_full_k{k_star}.csv", index=False, encoding="utf-8")
    export_cluster_sizes(sel, out_dir / f"cluster_sizes_k{k_star}.csv")
    export_cluster_means_anchor_long(sel, out_dir / f"cluster_means_anchor_long_k{k_star}.csv")
    export_gmm_params(gmm_star, out_dir / f"gmm_params_k{k_star}.csv")
    ms.to_csv(out_dir / "model_selection.csv", index=False, encoding="utf-8")

    summary = [
        ("input_csv", str(input_csv.name)),
        ("timestamp", time.strftime("%Y%m%d_%H%M%S")),
        ("n_complete", n_complete),
        ("prior_feasible", float(options.prior_feasible)),
        ("n_after_prior", n_after_prior),
        ("top_frac", float(options.top_frac)),
        ("rmse_threshold", float(threshold)),
        ("n_selected", n_selected),
        ("k_elbow", int(k_elbow)),
        ("k_candidates", ",".join(map(str, candidates)) if candidates else ""),
        ("k_star", int(k_star)),
        ("random_state", int(options.random_state)),
        ("k_select_rule", f"elbow+{int(options.elbow_plus)} then max bootstrap stability (ARI)"),
        ("umap_n_neighbors", int(options.n_neighbors)),
        ("umap_min_dist", float(options.min_dist)),
        ("umap_metric", str(options.metric)),
    ]
    pd.DataFrame(summary, columns=["key", "value"]).to_csv(out_dir / "run_summary.csv", index=False, encoding="utf-8")

    plt.figure()
    plt.plot(ms["k"], ms["bic"], marker="o")
    plt.axvline(k_elbow, linestyle="--", linewidth=1.0)
    plt.axvline(k_star, linestyle="--", linewidth=1.0)
    plt.xlabel("k")
    plt.ylabel("BIC")
    plt.title("GMM model selection in standardized CLR space")
    plt.savefig(out_dir / "fig_bic_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    mask = np.isfinite(ms["stability_mean"].to_numpy())
    x = ms.loc[mask, "k"].to_numpy(int)
    y = ms.loc[mask, "stability_mean"].to_numpy(float)
    yerr = ms.loc[mask, "stability_std"].to_numpy(float)
    if len(x):
        plt.errorbar(x, y, yerr=yerr, marker="o", capsize=3)
    plt.axvline(k_elbow, linestyle="--", linewidth=1.0)
    plt.axvline(k_star, linestyle="--", linewidth=1.0)
    plt.xlabel("k")
    plt.ylabel("Bootstrap stability (ARI)")
    plt.ylim([-0.05, 1.05])
    plt.title("Bootstrap stability in standardized CLR space")
    plt.savefig(out_dir / "fig_stability_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.scatter(sel["u1"], sel["u2"], c=sel["cluster"])
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title(f"UMAP visualization of CLR-GMM clusters (k={k_star})")
    plt.savefig(out_dir / f"fig_umap_clusters_k{k_star}.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.scatter(sel["f0"], sel["f1"], c=sel["cluster"])
    plt.xlabel("f0 (physics loss)")
    plt.ylabel("f1 (prior penalty)")
    plt.title(f"Objective space colored by CLR-GMM cluster (k={k_star})")
    plt.savefig(out_dir / f"fig_objective_clusters_k{k_star}.png", dpi=200, bbox_inches="tight")
    plt.close()

    counts = sel["cluster"].value_counts().sort_index()
    plt.figure()
    plt.bar(counts.index.astype(int), counts.values)
    plt.xlabel("cluster")
    plt.ylabel("count")
    plt.title(f"Cluster sizes (k={k_star})")
    plt.savefig(out_dir / f"fig_cluster_sizes_k{k_star}.png", dpi=200, bbox_inches="tight")
    plt.close()

    return out_dir

# ===== END INLINED MODULE: clustering.py =====


# ===== BEGIN ENTRYPOINT: run_stage2_clustering.py =====


import argparse
from pathlib import Path



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--prior-feasible", type=float, default=0.05)
    ap.add_argument("--use-prior-feasible", type=int, default=1)
    ap.add_argument("--top-frac", type=float, default=0.10)
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=12)
    ap.add_argument("--elbow-plus", type=int, default=2)
    ap.add_argument("--elbow-smooth", type=int, default=3)
    ap.add_argument("--random-state", type=int, default=0)
    ap.add_argument("--n-neighbors", type=int, default=40)
    ap.add_argument("--min-dist", type=float, default=0.05)
    ap.add_argument("--metric", type=str, default="euclidean")
    ap.add_argument("--n-boot", type=int, default=30)
    ap.add_argument("--subsample-frac", type=float, default=0.85)
    args = ap.parse_args()

    options = ClusteringOptions(
        prior_feasible=float(args.prior_feasible),
        use_prior_feasible=bool(int(args.use_prior_feasible)),
        top_frac=float(args.top_frac),
        k_min=int(args.k_min),
        k_max=int(args.k_max),
        elbow_plus=int(args.elbow_plus),
        elbow_smooth=int(args.elbow_smooth),
        random_state=int(args.random_state),
        n_neighbors=int(args.n_neighbors),
        min_dist=float(args.min_dist),
        metric=str(args.metric),
        n_boot=int(args.n_boot),
        subsample_frac=float(args.subsample_frac),
    )
    run_stage2_clustering(args.input_csv, args.out_dir, options)
    print(f"Done. Outputs written to: {args.out_dir}")


if __name__ == "__main__":
    main()

# ===== END ENTRYPOINT: run_stage2_clustering.py =====
