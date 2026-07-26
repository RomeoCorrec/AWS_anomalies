# Setup local MVTec AD / PatchCore Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the local, exploratory foundation for the MVTec AD / PatchCore anomaly detection project: repo scaffolding, dataset download/verification, an exploration notebook, and a working PatchCore training script on the `bottle` category.

**Architecture:** A `src/` package with focused modules (`config`, `data/download`, `models/train`, `eval/metrics`, `eval/log_results`), driven entirely by YAML configs under `config/`. Training uses anomalib's Python `Engine` API (`anomalib.data.MVTecAD`, `anomalib.models.Patchcore`, `anomalib.engine.Engine`). Every run appends its full flattened config + metrics to `results/experiments.csv`.

**Tech Stack:** Python 3.10+, `uv`, `anomalib` v2.x, `torch`/`torchvision` (CPU wheels), `omegaconf`, `pandas`/`csv` stdlib, `pytest`, Jupyter.

## Global Constraints

- Python 3.10+, dependency management via `uv` (`pyproject.toml`).
- `torch`/`torchvision` must resolve to CPU-only wheels (no CUDA download on this machine).
- No hardcoded paths, hyperparameters, or configs in Python source — everything lives in `config/*.yaml`.
- Code must be parameterizable by category (`bottle`, `screw`, `carpet`) and backbone (`wide_resnet50_2`, `dinov2_vits14`) from day one.
- `coreset_sampling_ratio` must be a config value (future ablation: 0.01 / 0.1 / 0.25).
- `data/` (dataset) and `results/*.csv` (experiment outputs) are gitignored.
- Every experiment run appends one row (full config, flattened, + metrics) to `results/experiments.csv` — never overwritten.
- Dataset download goes through anomalib's built-in downloader (`anomalib.data.MVTecAD`); on failure, raise an explicit error pointing to the HuggingFace mirror fallback — no silent retries.
- Type hints on public signatures; short one-line docstrings.
- Tests via `pytest`; no test hits the network or requires the real MVTec dataset.
- Decision threshold calibration is explicitly out of scope for this plan (reserved for the user).

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/__init__.py`
- Create: `src/data/__init__.py`
- Create: `src/models/__init__.py`
- Create: `src/eval/__init__.py`
- Create: `results/.gitkeep`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: an installable environment (`uv sync`) and an importable `src` package that later tasks build on (`src.config`, `src.data.download`, `src.models.train`, `src.eval.metrics`, `src.eval.log_results`).

- [ ] **Step 1: Create the directory skeleton**

```bash
mkdir -p config/data config/model config/experiment
mkdir -p src/data src/models src/eval
mkdir -p notebooks results tests
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "aws-anomalies"
version = "0.1.0"
description = "Détection d'anomalies visuelles non supervisée sur pièces industrielles (MVTec AD, PatchCore, anomalib)."
requires-python = ">=3.10"
dependencies = [
    "anomalib>=2.0.0",
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "omegaconf>=2.3.0",
]

[dependency-groups]
dev = [
    "pytest>=8.0.0",
    "jupyter>=1.0.0",
]

[tool.uv.sources]
torch = [{ index = "pytorch-cpu" }]
torchvision = [{ index = "pytorch-cpu" }]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.pytest.ini_options]
pythonpath = ["."]
```

- [ ] **Step 3: Write `.gitignore`**

```gitignore
.venv/
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
.ipynb_checkpoints/

data/

results/*.csv
!results/.gitkeep
```

- [ ] **Step 4: Create empty package markers and results placeholder**

```bash
touch src/__init__.py src/data/__init__.py src/models/__init__.py src/eval/__init__.py results/.gitkeep
```

- [ ] **Step 5: Install dependencies and verify**

Run: `uv sync`
Expected: environment created under `.venv/`, no errors.

Run: `uv run python -c "import torch, anomalib; print(torch.__version__, torch.version.cuda)"`
Expected: prints a torch version and `None` for `torch.version.cuda` (confirms CPU-only wheel).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore src/__init__.py src/data/__init__.py src/models/__init__.py src/eval/__init__.py results/.gitkeep uv.lock
git commit -m "chore: scaffold project with uv, CPU torch, package skeleton"
```

---

### Task 2: Config loader + YAML configs

**Files:**
- Create: `config/data/mvtec.yaml`
- Create: `config/model/patchcore.yaml`
- Create: `config/experiment/bottle_wideresnet50.yaml`
- Create: `src/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: `src` package from Task 1.
- Produces: `load_experiment_config(experiment_path: Path) -> DictConfig` — used by Task 5 (`src/models/train.py`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:

```python
from pathlib import Path

from omegaconf import OmegaConf

from src.config import load_experiment_config


def test_load_experiment_config_merges_and_overrides(tmp_path: Path) -> None:
    data_cfg = tmp_path / "data.yaml"
    data_cfg.write_text("root: data/mvtec\ncategory: bottle\nimage_size: [256, 256]\n")

    model_cfg = tmp_path / "model.yaml"
    model_cfg.write_text("backbone: wide_resnet50_2\ncoreset_sampling_ratio: 0.1\n")

    experiment_cfg = tmp_path / "experiment.yaml"
    experiment_cfg.write_text(
        f"data_config: {data_cfg}\n"
        f"model_config: {model_cfg}\n"
        "overrides:\n"
        "  category: screw\n"
    )

    cfg = load_experiment_config(experiment_cfg)

    assert cfg.category == "screw"
    assert cfg.backbone == "wide_resnet50_2"
    assert cfg.coreset_sampling_ratio == 0.1
    assert OmegaConf.to_container(cfg.image_size) == [256, 256]


def test_repo_bottle_experiment_config_loads() -> None:
    cfg = load_experiment_config(Path("config/experiment/bottle_wideresnet50.yaml"))

    assert cfg.category == "bottle"
    assert cfg.backbone == "wide_resnet50_2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.config'` (and the second test fails because `config/experiment/bottle_wideresnet50.yaml` doesn't exist yet).

- [ ] **Step 3: Write the YAML configs**

Create `config/data/mvtec.yaml`:

```yaml
root: "data/mvtec"
category: "bottle"
image_size: [256, 256]
train_batch_size: 32
eval_batch_size: 32
num_workers: 4
```

Create `config/model/patchcore.yaml`:

```yaml
backbone: "wide_resnet50_2"
layers: ["layer2", "layer3"]
coreset_sampling_ratio: 0.1
num_neighbors: 9
```

Create `config/experiment/bottle_wideresnet50.yaml`:

```yaml
data_config: "config/data/mvtec.yaml"
model_config: "config/model/patchcore.yaml"
overrides:
  category: "bottle"
```

- [ ] **Step 4: Write `src/config.py`**

```python
"""Chargement et fusion des configs YAML (data + model + overrides d'expérience)."""
from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_experiment_config(experiment_path: Path) -> DictConfig:
    """Fusionne data.yaml + model.yaml + les overrides du fichier d'expérience."""
    experiment_cfg = OmegaConf.load(experiment_path)

    data_cfg_path = Path(experiment_cfg.data_config)
    model_cfg_path = Path(experiment_cfg.model_config)

    base_cfg = OmegaConf.merge(
        OmegaConf.load(data_cfg_path),
        OmegaConf.load(model_cfg_path),
    )
    overrides = experiment_cfg.get("overrides", OmegaConf.create({}))
    return OmegaConf.merge(base_cfg, overrides)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add config/data/mvtec.yaml config/model/patchcore.yaml config/experiment/bottle_wideresnet50.yaml src/config.py tests/test_config.py
git commit -m "feat: add YAML config loader with data/model merge and per-experiment overrides"
```

---

### Task 3: Dataset download + verification

**Files:**
- Create: `src/data/download.py`
- Test: `tests/test_download.py`

**Interfaces:**
- Consumes: nothing beyond `anomalib.data.MVTecAD`.
- Produces: `download_category(category: str, root: Path) -> None`, `verify_category(category: str, root: Path) -> bool`, `MVTecDownloadError` — usable standalone via CLI (`python -m src.data.download --category bottle`); not consumed by other tasks in this plan, but this is the step the user runs before Task 4 and Task 5's manual verification.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_download.py`:

```python
from pathlib import Path

import pytest

from src.data.download import MVTecDownloadError, download_category, verify_category


def _make_valid_category(root: Path, category: str) -> None:
    cat_root = root / category
    (cat_root / "train" / "good").mkdir(parents=True)
    (cat_root / "test" / "good").mkdir(parents=True)
    (cat_root / "test" / "broken").mkdir(parents=True)
    (cat_root / "ground_truth" / "broken").mkdir(parents=True)


def test_verify_category_true_for_valid_structure(tmp_path: Path) -> None:
    _make_valid_category(tmp_path, "bottle")

    assert verify_category("bottle", tmp_path) is True


def test_verify_category_false_when_missing(tmp_path: Path) -> None:
    assert verify_category("bottle", tmp_path) is False


def test_verify_category_false_without_defect_dir(tmp_path: Path) -> None:
    cat_root = tmp_path / "bottle"
    (cat_root / "train" / "good").mkdir(parents=True)
    (cat_root / "test" / "good").mkdir(parents=True)
    (cat_root / "ground_truth").mkdir(parents=True)

    assert verify_category("bottle", tmp_path) is False


def test_download_category_wraps_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _FailingDataModule:
        def __init__(self, root: str, category: str) -> None:
            pass

        def prepare_data(self) -> None:
            raise RuntimeError("404")

    monkeypatch.setattr("src.data.download.MVTecAD", _FailingDataModule)

    with pytest.raises(MVTecDownloadError):
        download_category("bottle", tmp_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_download.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.download'`.

- [ ] **Step 3: Write `src/data/download.py`**

```python
"""Téléchargement et vérification de l'arborescence des catégories MVTec AD."""
from __future__ import annotations

import argparse
from pathlib import Path

from anomalib.data import MVTecAD

MVTEC_FALLBACK_URL = "https://huggingface.co/datasets/TheoM55/mvtec_all_objects_split"
EXPECTED_SPLIT_DIRS = ("train/good", "test/good")


class MVTecDownloadError(RuntimeError):
    """Levée quand le téléchargement anomalib échoue (ex: 404 serveur MVTec)."""


def download_category(category: str, root: Path) -> None:
    """Télécharge une catégorie MVTec AD via le datamodule anomalib si absente."""
    try:
        datamodule = MVTecAD(root=str(root), category=category)
        datamodule.prepare_data()
    except Exception as exc:
        raise MVTecDownloadError(
            f"Échec du téléchargement de la catégorie '{category}' via anomalib. "
            f"Télécharge-la manuellement depuis {MVTEC_FALLBACK_URL} et place-la "
            f"dans {root / category}."
        ) from exc


def verify_category(category: str, root: Path) -> bool:
    """Vérifie que l'arborescence attendue pour une catégorie MVTec AD est présente."""
    category_root = root / category
    if not category_root.is_dir():
        return False
    for split_dir in EXPECTED_SPLIT_DIRS:
        if not (category_root / split_dir).is_dir():
            return False
    test_dir = category_root / "test"
    defect_dirs = [d for d in test_dir.iterdir() if d.is_dir() and d.name != "good"]
    if not defect_dirs:
        return False
    if not (category_root / "ground_truth").is_dir():
        return False
    return True


def main() -> None:
    """CLI : télécharge puis vérifie une catégorie MVTec AD."""
    parser = argparse.ArgumentParser(description="Télécharge et vérifie une catégorie MVTec AD.")
    parser.add_argument("--category", required=True, help="bottle | screw | carpet")
    parser.add_argument("--root", default="data/mvtec", help="Racine du dataset MVTec AD")
    args = parser.parse_args()

    root = Path(args.root)
    download_category(args.category, root)
    if not verify_category(args.category, root):
        raise MVTecDownloadError(
            f"Arborescence invalide pour '{args.category}' après téléchargement dans {root}."
        )
    print(f"Catégorie '{args.category}' vérifiée dans {root / args.category}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_download.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/data/download.py tests/test_download.py
git commit -m "feat: add MVTec AD download wrapper and structure verification"
```

- [ ] **Step 6: Manual verification — download the real `bottle` category**

Run: `uv run python -m src.data.download --category bottle`
Expected: either `Catégorie 'bottle' vérifiée dans data/mvtec/bottle`, or an `MVTecDownloadError` naming the HuggingFace fallback (in which case, download manually and re-run to confirm `verify_category` now passes).

---

### Task 4: Exploration notebook

**Files:**
- Create: `notebooks/01_explore_dataset.ipynb`

**Interfaces:**
- Consumes: the `bottle` category downloaded to `data/mvtec/bottle` in Task 3's manual step.
- Produces: nothing consumed by other tasks — this is a terminal, human-facing deliverable.

- [ ] **Step 1: Create the notebook**

Create `notebooks/01_explore_dataset.ipynb`:

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# Exploration MVTec AD — catégorie `bottle`\n", "Change `CATEGORY` ci-dessous pour explorer une autre catégorie."]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from pathlib import Path\n",
    "from collections import Counter\n",
    "\n",
    "from PIL import Image\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "CATEGORY = \"bottle\"\n",
    "ROOT = Path(\"../data/mvtec\") / CATEGORY\n",
    "assert ROOT.exists(), f\"{ROOT} introuvable — lance d'abord: python -m src.data.download --category {CATEGORY}\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 1. Comptage d'images par split et par type de défaut"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "counts = Counter()\n",
    "for split_dir in (ROOT / \"train\").iterdir():\n",
    "    counts[f\"train/{split_dir.name}\"] = len(list(split_dir.glob(\"*.png\")))\n",
    "for split_dir in (ROOT / \"test\").iterdir():\n",
    "    counts[f\"test/{split_dir.name}\"] = len(list(split_dir.glob(\"*.png\")))\n",
    "for name, n in sorted(counts.items()):\n",
    "    print(f\"{name:30s} {n}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 2. Exemples : image normale vs défectueuse + masque"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "defect_types = [d.name for d in (ROOT / \"test\").iterdir() if d.name != \"good\"]\n",
    "defect_type = defect_types[0]\n",
    "\n",
    "good_img = next((ROOT / \"test\" / \"good\").glob(\"*.png\"))\n",
    "defect_img = next((ROOT / \"test\" / defect_type).glob(\"*.png\"))\n",
    "mask_img = ROOT / \"ground_truth\" / defect_type / f\"{defect_img.stem}_mask.png\"\n",
    "\n",
    "fig, axes = plt.subplots(1, 3, figsize=(12, 4))\n",
    "axes[0].imshow(Image.open(good_img)); axes[0].set_title(\"normale\")\n",
    "axes[1].imshow(Image.open(defect_img)); axes[1].set_title(f\"défaut: {defect_type}\")\n",
    "axes[2].imshow(Image.open(mask_img), cmap=\"gray\"); axes[2].set_title(\"masque ground truth\")\n",
    "for ax in axes: ax.axis(\"off\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 3. Distribution des tailles d'image"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "sizes = Counter()\n",
    "for img_path in ROOT.rglob(\"*.png\"):\n",
    "    with Image.open(img_path) as img:\n",
    "        sizes[img.size] += 1\n",
    "print(sizes)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 4. Sanity checks : doublons et extensions"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import hashlib\n",
    "\n",
    "extensions = Counter(p.suffix for p in ROOT.rglob(\"*\") if p.is_file())\n",
    "print(\"Extensions:\", extensions)\n",
    "\n",
    "hashes = Counter()\n",
    "for img_path in ROOT.rglob(\"*.png\"):\n",
    "    hashes[hashlib.md5(img_path.read_bytes()).hexdigest()] += 1\n",
    "duplicates = {h: n for h, n in hashes.items() if n > 1}\n",
    "print(f\"Doublons détectés: {len(duplicates)}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 2: Manual verification — run the notebook**

Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_explore_dataset.ipynb`
Expected: exits without error; reopening the notebook shows populated counts, the 3-panel image comparison, the size distribution, and a duplicates count (0 expected for MVTec AD).

- [ ] **Step 3: Commit**

```bash
git add notebooks/01_explore_dataset.ipynb
git commit -m "docs: add MVTec AD exploration notebook for the bottle category"
```

---

### Task 5: Eval module — metrics extraction + CSV logging

**Files:**
- Create: `src/eval/metrics.py`
- Create: `src/eval/log_results.py`
- Test: `tests/test_eval.py`

**Interfaces:**
- Consumes: nothing beyond `omegaconf.DictConfig`.
- Produces: `extract_metrics(test_results: list[dict[str, float]]) -> dict[str, float]` and `log_experiment(cfg: DictConfig, metrics: dict[str, float], csv_path: Path) -> None` — both consumed by Task 6 (`src/models/train.py`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_eval.py`:

```python
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.eval.log_results import log_experiment
from src.eval.metrics import extract_metrics


def test_extract_metrics_picks_known_keys() -> None:
    test_results = [{"image_AUROC": 0.987, "pixel_AUROC": 0.912, "unused_key": 1}]

    metrics = extract_metrics(test_results)

    assert metrics == {"image_AUROC": 0.987, "pixel_AUROC": 0.912}


def test_extract_metrics_raises_on_empty_results() -> None:
    with pytest.raises(ValueError):
        extract_metrics([])


def test_log_experiment_appends_row_with_header(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {"category": "bottle", "coreset_sampling_ratio": 0.1, "layers": ["layer2", "layer3"]}
    )
    metrics = {"image_AUROC": 0.98}
    csv_path = tmp_path / "experiments.csv"

    log_experiment(cfg, metrics, csv_path)
    log_experiment(cfg, metrics, csv_path)

    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    assert "category" in lines[0]
    assert "image_AUROC" in lines[0]
    assert "layer2,layer3" in lines[1]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_eval.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.eval.metrics'`.

- [ ] **Step 3: Write `src/eval/metrics.py`**

```python
"""Extraction des métriques utiles depuis les résultats de Engine.test()."""
from __future__ import annotations

METRIC_KEYS = ("image_AUROC", "image_F1Score", "pixel_AUROC")


def extract_metrics(test_results: list[dict[str, float]]) -> dict[str, float]:
    """Aplatit les résultats de Engine.test() en un dict des métriques suivies."""
    if not test_results:
        raise ValueError("test_results est vide, aucune métrique à extraire.")
    raw = test_results[0]
    return {key: float(raw[key]) for key in METRIC_KEYS if key in raw}
```

- [ ] **Step 4: Write `src/eval/log_results.py`**

```python
"""Écrit les résultats d'une run (config + métriques) dans un CSV d'expériences."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def log_experiment(cfg: DictConfig, metrics: dict[str, float], csv_path: Path) -> None:
    """Append une ligne (config aplatie + métriques) dans csv_path, en créant l'en-tête si besoin."""
    flat_cfg: dict[str, Any] = _flatten(OmegaConf.to_container(cfg, resolve=True))
    row = {**flat_cfg, **metrics}

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Aplatit un dict imbriqué en clés séparées par des points ; les listes sont jointes par virgule."""
    flat: dict[str, Any] = {}
    for key, value in d.items():
        full_key = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, prefix=f"{full_key}."))
        elif isinstance(value, list):
            flat[full_key] = ",".join(str(v) for v in value)
        else:
            flat[full_key] = value
    return flat
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_eval.py -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
git add src/eval/metrics.py src/eval/log_results.py tests/test_eval.py
git commit -m "feat: add metrics extraction and reproducible CSV experiment logging"
```

---

### Task 6: PatchCore training script

**Files:**
- Create: `src/models/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `load_experiment_config` (Task 2), `extract_metrics` + `log_experiment` (Task 5).
- Produces: `build_datamodule(cfg: DictConfig) -> MVTecAD`, `build_model(cfg: DictConfig) -> Patchcore`, `run_experiment(experiment_path: Path, results_csv: Path) -> dict[str, float]` — terminal deliverable of this plan, run manually end-to-end.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_train.py`:

```python
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from omegaconf import OmegaConf

from src.models.train import build_datamodule, build_model


def test_build_datamodule_constructs_without_error() -> None:
    cfg = OmegaConf.create(
        {
            "root": "data/mvtec",
            "category": "bottle",
            "train_batch_size": 16,
            "eval_batch_size": 8,
            "num_workers": 2,
        }
    )

    datamodule = build_datamodule(cfg)

    assert isinstance(datamodule, MVTecAD)


def test_build_model_constructs_without_error() -> None:
    cfg = OmegaConf.create(
        {
            "backbone": "wide_resnet50_2",
            "layers": ["layer2", "layer3"],
            "coreset_sampling_ratio": 0.1,
            "num_neighbors": 9,
        }
    )

    model = build_model(cfg)

    assert isinstance(model, Patchcore)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_train.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.models.train'`.

- [ ] **Step 3: Write `src/models/train.py`**

```python
"""Entraîne PatchCore sur une catégorie MVTec AD à partir d'une config d'expérience."""
from __future__ import annotations

import argparse
from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore
from omegaconf import DictConfig

from src.config import load_experiment_config
from src.eval.log_results import log_experiment
from src.eval.metrics import extract_metrics

DEFAULT_RESULTS_CSV = Path("results/experiments.csv")


def build_datamodule(cfg: DictConfig) -> MVTecAD:
    """Construit le datamodule MVTec AD à partir de la config fusionnée."""
    return MVTecAD(
        root=cfg.root,
        category=cfg.category,
        train_batch_size=cfg.train_batch_size,
        eval_batch_size=cfg.eval_batch_size,
        num_workers=cfg.num_workers,
    )


def build_model(cfg: DictConfig) -> Patchcore:
    """Construit le modèle PatchCore à partir de la config fusionnée."""
    return Patchcore(
        backbone=cfg.backbone,
        layers=list(cfg.layers),
        coreset_sampling_ratio=cfg.coreset_sampling_ratio,
        num_neighbors=cfg.num_neighbors,
    )


def run_experiment(experiment_path: Path, results_csv: Path = DEFAULT_RESULTS_CSV) -> dict[str, float]:
    """Charge la config, entraîne PatchCore, teste, et log les résultats en CSV."""
    cfg = load_experiment_config(experiment_path)
    datamodule = build_datamodule(cfg)
    model = build_model(cfg)

    engine = Engine()
    engine.fit(datamodule=datamodule, model=model)
    test_results = engine.test(datamodule=datamodule, model=model)

    metrics = extract_metrics(test_results)
    log_experiment(cfg, metrics, results_csv)
    return metrics


def main() -> None:
    """CLI : entraîne et évalue PatchCore pour une config d'expérience donnée."""
    parser = argparse.ArgumentParser(description="Entraîne PatchCore sur MVTec AD.")
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--results-csv", default=DEFAULT_RESULTS_CSV, type=Path)
    args = parser.parse_args()

    metrics = run_experiment(args.experiment, args.results_csv)
    print(f"Résultats : {metrics}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_train.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/models/train.py tests/test_train.py
git commit -m "feat: add PatchCore training script using anomalib Engine API"
```

- [ ] **Step 6: Manual verification — end-to-end run on `bottle`**

Run: `uv run python -m src.models.train --experiment config/experiment/bottle_wideresnet50.yaml`
Expected: training/coreset logs from anomalib, then `Résultats : {...}` printed with `image_AUROC` (and other configured metrics); `results/experiments.csv` now contains a header row plus one data row with the full flattened `bottle`/`wide_resnet50_2` config and those metrics.

---

## Self-Review Notes

- **Spec coverage:** repo structure (Task 1), config YAML (Task 2), download + verification (Task 3), notebook (Task 4), CSV logging (Task 5), PatchCore training via Engine API (Task 6) — all five original scope items plus reproducible logging are covered.
- **Placeholder scan:** no TBD/TODO; every step has literal code or an exact command with an expected outcome.
- **Type consistency:** `load_experiment_config` (Task 2) returns `DictConfig`, consumed as `cfg` by `build_datamodule`/`build_model` (Task 6); `extract_metrics` returns `dict[str, float]`, consumed by `log_experiment` (Task 5) and by `run_experiment` (Task 6) — signatures match across tasks.
