"""Extract the bundled SQLite databases used by Spider Dev evaluation."""

import json
import shutil
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

DATA_DIR = Path(__file__).resolve().parent


def _same_contents(target: Path, archive: ZipFile, member: str) -> bool:
    with target.open("rb") as existing, archive.open(member, "r") as bundled:
        while True:
            existing_chunk = existing.read(1024 * 1024)
            bundled_chunk = bundled.read(1024 * 1024)
            if existing_chunk != bundled_chunk:
                return False
            if not existing_chunk:
                return True


def extract_dev_databases(data_dir: Path = DATA_DIR) -> int:
    data_dir = Path(data_dir)
    dev_path = data_dir / "spider_dev.json"
    archive_path = data_dir / "spider_dev_databases.zip"
    if not dev_path.is_file():
        raise FileNotFoundError(f"Spider Dev data not found: {dev_path}")
    if not archive_path.is_file():
        raise FileNotFoundError(f"Spider Dev database archive not found: {archive_path}")

    try:
        with dev_path.open("r", encoding="utf-8") as file:
            dev_examples = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read Spider Dev data {dev_path}: {error}") from error
    if not isinstance(dev_examples, list) or not dev_examples:
        raise ValueError(f"Spider Dev data must be a non-empty JSON list: {dev_path}")

    if any(not isinstance(example, dict) for example in dev_examples):
        raise ValueError("Every Spider Dev example must be a JSON object.")
    raw_dev_ids = [example.get("db_id") for example in dev_examples]
    if any(not isinstance(db_id, str) or not db_id for db_id in raw_dev_ids):
        raise ValueError("Spider Dev examples must all contain a non-empty db_id.")
    dev_ids = set(raw_dev_ids)

    database_root = data_dir / "database"
    extracted_ids = set()
    with ZipFile(archive_path, "r") as archive:
        members = [
            member
            for member in archive.infolist()
            if not member.is_dir() and PurePosixPath(member.filename).suffix == ".sqlite"
        ]
        if len(members) != len(dev_ids):
            raise ValueError(
                f"Expected {len(dev_ids)} SQLite files in {archive_path}, "
                f"found {len(members)}."
            )

        for member in members:
            parts = PurePosixPath(member.filename).parts
            if len(parts) != 3 or parts[0] != "database":
                raise ValueError(f"Unexpected path in Dev database archive: {member.filename}")
            db_id = parts[1]
            if db_id not in dev_ids or parts[2] != f"{db_id}.sqlite":
                raise ValueError(f"Unexpected database in Dev archive: {member.filename}")
            if db_id in extracted_ids:
                raise ValueError(f"Duplicate database in Dev archive: {db_id}")
            extracted_ids.add(db_id)

            target = database_root / db_id / f"{db_id}.sqlite"
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if not _same_contents(target, archive, member.filename):
                    raise FileExistsError(
                        f"Existing database differs from bundled Dev data: {target}"
                    )
                continue

            temporary = target.with_name(target.name + ".part")
            with archive.open(member, "r") as source, temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            temporary.replace(target)

    missing = sorted(
        db_id
        for db_id in dev_ids
        if not (database_root / db_id / f"{db_id}.sqlite").is_file()
    )
    if missing:
        raise FileNotFoundError(
            "Spider Dev databases are incomplete after extraction: "
            + ", ".join(missing)
        )
    return len(dev_ids)


def main() -> None:
    count = extract_dev_databases()
    print(f"Ready: {count} Spider Dev SQLite databases in {DATA_DIR / 'database'}.")


if __name__ == "__main__":
    main()
