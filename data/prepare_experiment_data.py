"""Prepare leakage-filtered SQL, CoT, and repair-format SFT examples."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.execution import execute_readonly_query
from src.prompts import (
    prompt_generator_cot,
    prompt_m1_zeroshot,
    prompt_naive_repair,
    synthesize_cot_reasoning,
)
from src.schema_engine import extract_schema_ddl, locate_database_path

DATA_DIR = PROJECT_ROOT / "data"
DATABASE_DIR = DATA_DIR / "database"
OUTPUT_DIR = DATA_DIR / "processed"
FORMATS = ("sql", "cot", "repair")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def make_repair_example(sample: Dict[str, str], schema: str, db_path: Path) -> Dict[str, str]:
    gold_sql = sample["query"].strip().rstrip(";")
    failed_sql = f"{gold_sql}\n__repair_syntax_error__"
    success, result = execute_readonly_query(db_path, failed_sql)
    if success:
        failed_sql = "SELECT FROM __repair_syntax_error__"
        success, result = execute_readonly_query(db_path, failed_sql)
    if success:
        raise RuntimeError(f"Could not synthesize a failing query for {sample['db_id']}")

    return {
        "input_text": prompt_naive_repair(
            schema, sample["question"], failed_sql, str(result)
        ),
        "target_text": f"```sql\n{gold_sql}\n```",
    }


def prepare(format_name: str, sample_count: int) -> Path:
    train_path = DATA_DIR / "train_spider.json"
    dev_path = DATA_DIR / "spider_dev.json"
    realistic_path = DATA_DIR / "spider_realistic.json"
    missing = [path for path in (train_path, dev_path, realistic_path) if not path.is_file()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Missing Spider data: {missing_text}. Run data/download_and_prepare.py first."
        )

    excluded_db_ids = {
        sample["db_id"]
        for path in (dev_path, realistic_path)
        for sample in load_json(path)
    }
    candidates = [
        sample for sample in load_json(train_path)
        if sample["db_id"] not in excluded_db_ids
    ]

    examples: List[Dict[str, str]] = []
    schema_cache: Dict[str, str] = {}
    for sample in candidates:
        if len(examples) >= sample_count:
            break
        db_id = sample["db_id"]
        db_path = locate_database_path(db_id, DATABASE_DIR)
        if not db_path.is_file():
            continue

        schema = schema_cache.get(db_id)
        if schema is None:
            schema = extract_schema_ddl(db_path)
            schema_cache[db_id] = schema
        if not schema:
            continue

        if format_name == "sql":
            input_text = prompt_m1_zeroshot(schema, sample["question"])
            target_text = f"```sql\n{sample['query'].strip()}\n```"
        elif format_name == "cot":
            input_text = prompt_generator_cot(schema, sample["question"])
            target_text = synthesize_cot_reasoning(
                sample["question"], sample["query"], schema
            )
        else:
            repair_example = make_repair_example(sample, schema, db_path)
            input_text = repair_example["input_text"]
            target_text = repair_example["target_text"]

        examples.append({
            "db_id": db_id,
            "question": sample["question"],
            "schema": schema,
            "input_text": input_text,
            "target_text": target_text,
        })

    if len(examples) < sample_count:
        raise RuntimeError(
            f"Prepared {len(examples)} of {sample_count} requested examples. "
            "Check that Spider train data and its SQLite databases are complete."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{format_name}_train_{sample_count}.json"
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(examples, file, ensure_ascii=False, indent=2)
    temporary_path.replace(output_path)
    print(f"Wrote {len(examples)} leakage-filtered examples to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=FORMATS, required=True)
    parser.add_argument("--samples", type=int, default=800)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    prepare(args.format, args.samples)


if __name__ == "__main__":
    main()
