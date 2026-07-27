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


class _FailingDataModule:
    def __init__(self, root: str, category: str) -> None:
        pass

    def prepare_data(self) -> None:
        raise RuntimeError("404")


def test_download_category_wraps_errors_when_both_sources_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _failing_hf_download(category: str, root: Path) -> None:
        raise RuntimeError("network unavailable")

    monkeypatch.setattr("src.data.download.MVTecAD", _FailingDataModule)
    monkeypatch.setattr("src.data.download._download_from_huggingface", _failing_hf_download)

    with pytest.raises(MVTecDownloadError):
        download_category("bottle", tmp_path)


def test_download_category_falls_back_to_huggingface_on_anomalib_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def _fake_hf_download(category: str, root: Path) -> None:
        calls.append(category)

    monkeypatch.setattr("src.data.download.MVTecAD", _FailingDataModule)
    monkeypatch.setattr("src.data.download._download_from_huggingface", _fake_hf_download)

    download_category("bottle", tmp_path)

    assert calls == ["bottle"]
