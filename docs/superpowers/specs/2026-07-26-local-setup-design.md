# Setup local — détection d'anomalies visuelles (MVTec AD / PatchCore / anomalib)

Date : 2026-07-26

## Contexte

Portfolio technique de détection d'anomalies visuelles non supervisée sur pièces
industrielles (dataset MVTec AD), via PatchCore (librairie `anomalib`). Cette session
couvre uniquement la mise en place locale et exploratoire : structure du repo,
environnement, téléchargement/vérification du dataset, notebook d'exploration, et une
première config PatchCore fonctionnelle sur la catégorie `bottle`. Pas de code AWS, pas
de Docker, pas de Terraform à ce stade.

Catégories cibles à terme : `bottle`, `screw`, `carpet`. Backbones à comparer :
WideResNet50 (défaut) et DINOv2 ViT-S. Ablation prévue sur `coreset_sampling_ratio`
(0.01 / 0.1 / 0.25).

## Décisions retenues

- **Gestionnaire de dépendances** : `uv`, avec `pyproject.toml` (Python 3.10+), groupe
  `dev` pour pytest/jupyter. `torch`/`torchvision` en variante CPU explicite.
- **Stockage dataset** : `data/mvtec/<categorie>/` à la racine du repo, exclu de git via
  `.gitignore`.
- **Téléchargement dataset** : via le downloader intégré d'`anomalib`. Le serveur MVTec
  officiel a connu des 404 intermittents début 2026 — en cas d'échec, le script lève une
  erreur explicite pointant vers un fallback manuel (mirroir HuggingFace), sans retry
  automatique silencieux.
- **Notebooks** : Jupyter classique (`.ipynb`), pas de jupytext.
- **Tests** : `pytest`.
- **Interface d'entraînement** : script Python utilisant l'API `Engine` d'anomalib
  (v2.x) — `anomalib.data.MVTec`, `anomalib.models.Patchcore`, `anomalib.engine.Engine`
  — plutôt que la CLI anomalib, pour garder le contrôle sur le logging CSV des
  expériences et réutiliser ce code tel quel côté SageMaker plus tard.
- **Reproductibilité** : chaque run écrit ses métriques et sa config complète (aplatie)
  en une ligne dans `results/experiments.csv`.

## Architecture du repo

```
config/
  data/mvtec.yaml                       # racine dataset, catégorie, image_size, batch sizes
  model/patchcore.yaml                  # backbone, layers, coreset_sampling_ratio, num_neighbors
  experiment/bottle_wideresnet50.yaml   # combine data + model, override category, point d'entrée d'une run
src/
  data/
    download.py                         # download_category() + verify_category() (wrapper anomalib)
  models/
    train.py                            # charge config, construit datamodule/model/engine, fit+test
  eval/
    metrics.py                          # extraction des métriques depuis les résultats Engine.test()
    log_results.py                      # append métriques + config complète dans results/experiments.csv
notebooks/
  01_explore_dataset.ipynb              # exploration : comptage, exemples, masques, sanity checks
results/                                # CSV d'expériences (gitignored sauf .gitkeep)
data/                                   # dataset MVTec (gitignored)
tests/
  test_download.py
  test_config.py
pyproject.toml
.gitignore
```

## Détail par composant

### Config YAML

`config/data/mvtec.yaml` :
```yaml
root: "data/mvtec"
category: "bottle"        # bottle | screw | carpet
image_size: [256, 256]
train_batch_size: 32
eval_batch_size: 32
num_workers: 4
```

`config/model/patchcore.yaml` :
```yaml
backbone: "wide_resnet50_2"   # wide_resnet50_2 | dinov2_vits14
layers: ["layer2", "layer3"]  # ignoré si backbone = dinov2 (features globales)
coreset_sampling_ratio: 0.1
num_neighbors: 9
```

`config/experiment/bottle_wideresnet50.yaml` combine les deux configs de base et
surcharge `category` si besoin. Une run = un seul fichier expérience passé au script
d'entraînement (fusion via `OmegaConf.merge` ou équivalent). Aucune valeur en dur dans
`train.py`.

### Téléchargement + vérification (`src/data/download.py`)

- `download_category(category: str, root: Path) -> None` : appelle le downloader intégré
  d'anomalib pour la catégorie donnée.
- `verify_category(category: str, root: Path) -> bool` : vérifie l'arborescence attendue
  (`train/good`, `test/good`, `test/<defect_type>/`, `ground_truth/`) et un nombre
  d'images cohérent avec les specs connues de MVTec AD pour cette catégorie.
- En cas d'échec du download (404), erreur explicite avec pointeur vers le mirroir
  HuggingFace comme fallback manuel.
- CLI minimale : `python -m src.data.download --category bottle`.

### Notebook d'exploration (`notebooks/01_explore_dataset.ipynb`)

1. Chargement de la config YAML de la catégorie (paramétrable en haut du notebook).
2. Comptage d'images par split (train/test) et par type de défaut.
3. Affichage d'exemples : image normale vs défectueuse + masque ground truth superposé.
4. Distribution des tailles d'image, sanity check (doublons, extensions).

Aucune logique métier dans le notebook — uniquement lecture/visualisation ; tout calcul
réutilisable reste dans `src/`.

### Entraînement PatchCore (`src/models/train.py`)

1. Charger et fusionner les YAML (`data/mvtec.yaml` + `model/patchcore.yaml` + override
   d'expérience).
2. Construire le `datamodule` (`anomalib.data.MVTec`) avec `category`, `image_size`,
   `batch_size` depuis la config.
3. Construire le modèle `anomalib.models.Patchcore` avec `backbone`, `layers`,
   `coreset_sampling_ratio`, `num_neighbors` depuis la config.
4. `Engine(...).fit(datamodule=datamodule, model=model)` puis `.test(...)` pour récupérer
   les métriques.
5. Passer métriques + config complète (aplatie) à `src/eval/log_results.py`, qui les
   append dans `results/experiments.csv` (une ligne par run, colonnes = tous les
   hyperparamètres + métriques).

CLI : `python -m src.models.train --experiment config/experiment/bottle_wideresnet50.yaml`.

## Hors périmètre (sessions suivantes)

- Ablation backbone / coreset_sampling_ratio (utilisera la même config paramétrable).
- Calibration du seuil de décision (réservé à l'utilisateur, cf. CLAUDE.md).
- Packaging inférence local, Docker, S3, SageMaker, Lambda, API Gateway, Terraform.

## Tests prévus

- `tests/test_config.py` : la fusion des YAML produit bien la config attendue
  (override de catégorie, valeurs par défaut).
- `tests/test_download.py` : `verify_category` détecte correctement une arborescence
  valide vs incomplète (sur données factices, pas le vrai dataset).
