"""Download Spider, Spider-Realistic, and their SQLite databases."""

import json
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List

DATA_DIR = Path(__file__).resolve().parent
SPIDER_URL = "https://raw.githubusercontent.com/Yale-LILY/Spider/master/data/spider.zip"
SPIDER_REALISTIC_URL = (
    "https://huggingface.co/datasets/aherntech/spider-realistic/raw/main/"
    "spider-realistic.json"
)


def download_file(url: str, dest_path: Path) -> None:
    """Download atomically so an interrupted transfer is never mistaken for a dataset."""
    temporary_path = dest_path.with_suffix(dest_path.suffix + ".part")

    def progress_hook(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(downloaded / total_size * 100, 100.0)
            sys.stdout.write(
                f"\rDownloading: {percent:.1f}% "
                f"({downloaded / (1024 * 1024):.1f} MB)"
            )
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, temporary_path, reporthook=progress_hook)
        temporary_path.replace(dest_path)
        print()
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _find_path(root: Path, name: str) -> Path:
    matches = [
        path for path in root.rglob(name)
        if (path.is_dir() if name == "database" else path.is_file())
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {name!r} in the downloaded Spider archive; "
            f"found {len(matches)}."
        )
    return matches[0]


def _install_database(source: Path, destination: Path) -> None:
    if not destination.exists():
        source.rename(destination)
        return

    destination.mkdir(parents=True, exist_ok=True)
    for source_database in source.iterdir():
        target_database = destination / source_database.name
        if not target_database.exists():
            shutil.copytree(source_database, target_database)
            continue
        for source_file in source_database.rglob("*"):
            if not source_file.is_file():
                continue
            target_file = target_database / source_file.relative_to(source_database)
            if not target_file.exists():
                target_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, target_file)


def _sync_spider_data(force: bool = False) -> None:
    spider_zip = DATA_DIR / "spider.zip"
    database_dir = DATA_DIR / "database"
    core_files = (
        DATA_DIR / "spider_dev.json",
        DATA_DIR / "train_spider.json",
    )
    if not force and database_dir.is_dir() and all(path.is_file() for path in core_files):
        return

    if not spider_zip.is_file():
        print("Downloading the Spider archive (includes SQLite databases)...")
        download_file(SPIDER_URL, spider_zip)

    with tempfile.TemporaryDirectory(prefix="spider_extract_", dir=str(DATA_DIR)) as temp:
        extract_dir = Path(temp)
        try:
            with zipfile.ZipFile(spider_zip, "r") as archive:
                archive.extractall(extract_dir)
        except (OSError, zipfile.BadZipFile) as error:
            raise RuntimeError(
                f"Could not extract Spider archive {spider_zip}: {error}"
            ) from error

        source_database = _find_path(extract_dir, "database")
        source_dev = _find_path(extract_dir, "dev.json")
        source_train = _find_path(extract_dir, "train_spider.json")
        _install_database(source_database, database_dir)
        for source_file, destination in (
            (source_dev, DATA_DIR / "spider_dev.json"),
            (source_train, DATA_DIR / "train_spider.json"),
        ):
            if not destination.exists():
                shutil.copy2(source_file, destination)

    spider_zip.unlink(missing_ok=True)


def _load_examples(path: Path, required_fields: tuple) -> List[Dict[str, object]]:
    try:
        with path.open("r", encoding="utf-8") as file:
            examples = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read dataset {path}: {error}") from error
    if not isinstance(examples, list) or not examples:
        raise ValueError(f"Dataset {path} must contain a non-empty JSON list")
    for index, example in enumerate(examples):
        if not isinstance(example, dict) or any(
            field not in example for field in required_fields
        ):
            raise ValueError(
                f"Dataset {path} has an invalid record at index {index}; "
                f"expected fields {required_fields}"
            )
    return examples


def _missing_database_ids(examples: List[Dict[str, object]]) -> List[str]:
    database_dir = DATA_DIR / "database"
    return sorted({
        str(example["db_id"])
        for example in examples
        if not (database_dir / str(example["db_id"]) / f"{example['db_id']}.sqlite").is_file()
    })


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    realistic_path = DATA_DIR / "spider_realistic.json"
    if not realistic_path.is_file():
        print("Downloading Spider-Realistic...")
        download_file(SPIDER_REALISTIC_URL, realistic_path)

    _sync_spider_data()
    dev_examples = _load_examples(
        DATA_DIR / "spider_dev.json", ("db_id", "question", "query")
    )
    train_examples = _load_examples(
        DATA_DIR / "train_spider.json", ("db_id", "question", "query")
    )
    realistic_examples = _load_examples(
        realistic_path, ("db_id", "question", "query")
    )

    missing_ids = _missing_database_ids(dev_examples + realistic_examples)
    if missing_ids:
        print(
            f"Spider database folder is incomplete ({len(missing_ids)} evaluation "
            "databases missing); refreshing it from the Spider archive..."
        )
        _sync_spider_data(force=True)
        missing_ids = _missing_database_ids(dev_examples + realistic_examples)
    if missing_ids:
        preview = ", ".join(missing_ids[:10])
        if len(missing_ids) > 10:
            preview += ", ..."
        raise FileNotFoundError(
            f"Missing SQLite databases for: {preview}. "
            "Check the Spider archive and data/database folder."
        )

    print(
        f"Ready: {len(train_examples)} Spider train examples, "
        f"{len(dev_examples)} Spider Dev examples, "
        f"{len(realistic_examples)} Spider-Realistic examples, "
        f"{len({sample['db_id'] for sample in dev_examples + realistic_examples})} "
        "evaluation databases."
    )


if __name__ == "__main__":
    main()
