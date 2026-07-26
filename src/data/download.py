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
