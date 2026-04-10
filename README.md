# MU-SHOT-Fi: Self-Supervised Multi-User Wi-Fi Sensing with Source-Free Unsupervised Domain Adaptation

**Ahmed Y. Radwan** (Graduate Student Member, IEEE) and **Hina Tabassum** (Senior Member, IEEE), Department of Electrical Engineering and Computer Science, York University, Toronto, ON, Canada.

This repository holds reference implementations and experiment code for **WiDAR** (single-user, **SU-SHOT-Fi**) and **WiMANS** (multi-user, **MU-SHOT-Fi**) under a source-free UDA setting with optional self-supervised components (e.g. rotation SSL, CPC where applicable).

---

## Repository layout

| Path | Role |
|------|------|
| `WiDAR/` | **SU-SHOT-Fi** — WiDAR gesture CSI; domain adaptation across rooms / setups. |
| `WiDAR/SHOT/` | Baseline SHOT-style pipeline for WiDAR. |
| `WiDAR/SHOTPlus/` | Main **SU-SHOT-Fi** entry point: SHOT + rotation SSL and related options (`uda_widar.py`). |
| `WiDAR/SHOTPlus_CPC/` | SHOT + CPC extension for WiDAR (`uda_widar.py`). |
| `WiMANS/` | **MU-SHOT-Fi** — WiMANS multi-user activity CSI; occupancy-aware adaptation. |
| `WiMANS/SHOT/` | Legacy minimal SHOT WiMANS baseline. |
| `WiMANS/SHOTPlus/` | Main **MU-SHOT-Fi** entry point (`uda_wimans.py`). |
| `WiMANS/SHOTPlus_CPC/` | SHOT + CPC for WiMANS (`uda_wimans.py`). |

Other subpackages under `WiDAR/` or `WiMANS/` (if present) are experimental variants.

---

## Requirements

- Python 3.x, **CUDA** (training scripts expect a GPU).
- PyTorch and common scientific stack (e.g. `numpy`, `scipy`, `pandas`, `scikit-learn`, `tqdm`). Install versions compatible with your CUDA driver.

Datasets are **not** included. Place WiDAR and WiMANS data on your machine and point the code at them (see below).

---

## Data paths

Paths are resolved from each package’s `preset.py` using environment variables (defaults assume datasets sit next to the project parent directory):

| Variable | Typical role |
|----------|----------------|
| `WIDAR_DATA_ROOT` | Root of the organized WiDAR dataset (directory that contains room/user CSI layout expected by `load_data`). |
| `WIMANS_DATA_ROOT` | Root of the WiMANS dataset (directory that contains `wifi_csi/` and `annotation.csv`). |

If unset, defaults follow: `<parent-of-this-repo>/widar_dataset/organized_dataset` and `<parent-of-this-repo>/wimans_dataset`.

---

## How to run

From the repository root, `cd` into the variant you need, then launch the driver script. **Edit `preset.py` in that folder** for tasks, domains, hyperparameters, and `save_dir` before long runs.

### SU-SHOT-Fi (WiDAR)

```bash
export WIDAR_DATA_ROOT=/path/to/widar_dataset/organized_dataset   # optional if default layout matches
cd WiDAR/SHOTPlus
python uda_widar.py
```

CPC line:

```bash
cd WiDAR/SHOTPlus_CPC
python uda_widar.py
```

### MU-SHOT-Fi (WiMANS)

```bash
export WIMANS_DATA_ROOT=/path/to/wimans_dataset   # optional if default layout matches
cd WiMANS/SHOTPlus
python uda_wimans.py
```

CPC line:

```bash
cd WiMANS/SHOTPlus_CPC
python uda_wimans.py
```

Logs, checkpoints, and aggregated metrics are written under the `save_dir` defined in each `preset.py` (see `.gitignore` for patterns that stay local and are not intended for git).
