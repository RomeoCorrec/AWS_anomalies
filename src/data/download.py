"""Téléchargement et vérification de l'arborescence des catégories MVTec AD."""
from __future__ import annotations

import argparse
from pathlib import Path

from anomalib.data import MVTecAD

MVTEC_FALLBACK_DATASET = "TheoM55/mvtec_all_objects_split"
MVTEC_FALLBACK_URL = f"https://huggingface.co/datasets/{MVTEC_FALLBACK_DATASET}"
EXPECTED_SPLIT_DIRS = ("train/good", "test/good")


class MVTecDownloadError(RuntimeError):
    """Levée quand le téléchargement anomalib et le fallback HuggingFace échouent tous les deux."""


def download_category(category: str, root: Path) -> None:
    """Télécharge une catégorie MVTec AD via anomalib, ou via le mirroir HuggingFace si ça échoue."""
    try:
        datamodule = MVTecAD(root=str(root), category=category)
        datamodule.prepare_data()
        print(f"Catégorie '{category}' téléchargée via le serveur MVTec (anomalib).")
    except Exception as anomalib_exc:
        print(
            f"Échec du téléchargement de '{category}' via anomalib ({anomalib_exc}); "
            f"tentative via le mirroir HuggingFace {MVTEC_FALLBACK_DATASET}..."
        )
        try:
            _download_from_huggingface(category, root)
        except Exception as hf_exc:
            raise MVTecDownloadError(
                f"Échec du téléchargement de la catégorie '{category}' via anomalib et via le "
                f"mirroir HuggingFace {MVTEC_FALLBACK_DATASET}. Télécharge-la manuellement depuis "
                f"{MVTEC_FALLBACK_URL} et place-la dans {root / category}."
            ) from hf_exc
        print(f"Catégorie '{category}' téléchargée via le mirroir HuggingFace {MVTEC_FALLBACK_DATASET}.")


def _download_from_huggingface(category: str, root: Path) -> None:
    """Reconstruit l'arborescence MVTec AD d'une catégorie depuis le mirroir HuggingFace.

    Télécharge uniquement les shards Parquet de la catégorie demandée (pas le dataset
    entier, qui combine les 15 catégories MVTec) via huggingface_hub, puis décode les
    images/masques (stockés en octets bruts dans le Parquet) avec PIL.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files

    category_root = root / category
    all_files = list_repo_files(MVTEC_FALLBACK_DATASET, repo_type="dataset")

    for split in ("train", "test"):
        shard_files = sorted(f for f in all_files if f.startswith(f"data/{category}.{split}-"))
        if not shard_files:
            raise MVTecDownloadError(
                f"Aucun shard '{category}.{split}' trouvé sur {MVTEC_FALLBACK_DATASET}."
            )
        local_paths = [
            hf_hub_download(MVTEC_FALLBACK_DATASET, filename=shard, repo_type="dataset")
            for shard in shard_files
        ]
        table = pq.read_table(local_paths)
        _write_split(category_root, table)


def _write_split(category_root: Path, table: "pq.Table") -> None:
    """Écrit les images/masques d'un split (train ou test) dans l'arborescence MVTec AD."""
    from io import BytesIO

    from PIL import Image

    defect_counters: dict[str, int] = {}
    for row in table.to_pylist():
        split = row["split"]
        defect = row["defect"]
        index = defect_counters.get((split, defect), 0)
        defect_counters[(split, defect)] = index + 1

        image_dir = category_root / split / defect
        image_dir.mkdir(parents=True, exist_ok=True)
        Image.open(BytesIO(row["image_path"]["bytes"])).convert("RGB").save(
            image_dir / f"{index:03d}.png"
        )

        if row["mask_path"] is not None:
            mask_dir = category_root / "ground_truth" / defect
            mask_dir.mkdir(parents=True, exist_ok=True)
            Image.open(BytesIO(row["mask_path"]["bytes"])).convert("L").save(
                mask_dir / f"{index:03d}_mask.png"
            )


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
