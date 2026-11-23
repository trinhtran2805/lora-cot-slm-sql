"""Evaluate the paper's Dev-table logs with the official Spider evaluator."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "artifacts" / "results_json"
DEV_EXPERIMENTS = (
    "zero_shot_dev",
    "lora_sql_dev",
    "zero_shot_cot_dev",
    "lora_cot_naive_repair_dev",
    "lora_sql_naive_repair_dev",
    "lora_cot_dev",
    "zero_shot_cot_repair_dev",
    "lora_cot_repair_dev",
)
EXPERIMENT_LOGS = {
    experiment: Path("missing_experiments") / f"{experiment}.json"
    for experiment in DEV_EXPERIMENTS
}
EXPERIMENT_LOGS["lora_sql_cot_repair_dev"] = (
    Path("lora_sql_cot_repair_dev_full") / "lora_sql_cot_repair_dev.json"
)
PAPER_EXPERIMENT_ORDER = (
    "zero_shot_dev",
    "lora_sql_dev",
    "zero_shot_cot_dev",
    "lora_cot_naive_repair_dev",
    "lora_sql_naive_repair_dev",
    "lora_sql_cot_repair_dev",
    "lora_cot_dev",
    "zero_shot_cot_repair_dev",
    "lora_cot_repair_dev",
)


def one_line(value: object) -> str:
    return str(value or "").replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


def write_evaluation_inputs(
    experiment: str,
    dev_examples: List[Dict[str, object]],
    records: List[Dict[str, object]],
    output_dir: Path,
) -> Tuple[Path, Path]:
    if len(records) != len(dev_examples):
        raise ValueError(
            f"{experiment}: expected {len(dev_examples)} records, found {len(records)}."
        )

    gold_lines = []
    prediction_lines = []
    for index, (record, example) in enumerate(zip(records, dev_examples)):
        if record.get("index") != index or record.get("db_id") != example.get("db_id"):
            raise ValueError(f"{experiment}: record order mismatch at Dev row {index}.")

        gold_sql = one_line(example.get("query") or example.get("sql"))
        if not gold_sql:
            raise ValueError(f"{experiment}: no gold SQL at Dev row {index}.")
        gold_lines.append(f"{gold_sql}\t{example['db_id']}")

        if "repair_attempted" not in record:
            raise ValueError(
                f"{experiment}: repair_attempted is missing at Dev row {index}."
            )
        if record["repair_attempted"]:
            prediction = record.get("repair_sql")
        else:
            prediction = record.get("initial_sql")
        prediction_lines.append(one_line(prediction) or "__INVALID_SQL__")

    output_dir.mkdir(parents=True, exist_ok=True)
    gold_path = output_dir / "gold_dev.txt"
    prediction_path = output_dir / f"{experiment}_pred.txt"
    gold_path.write_text("\n".join(gold_lines) + "\n", encoding="utf-8")
    prediction_path.write_text(
        "\n".join(prediction_lines) + "\n", encoding="utf-8"
    )
    return gold_path, prediction_path


def parse_spider_scores(output: str) -> Tuple[float, float]:
    scores = {}
    for line in output.splitlines():
        fields = line.split()
        if fields and fields[0] == "execution":
            scores["execution"] = float(fields[-1])
        elif len(fields) > 1 and fields[:2] == ["exact", "match"]:
            scores["exact_match"] = float(fields[-1])
    missing = {"execution", "exact_match"} - scores.keys()
    if missing:
        raise ValueError(
            "Spider evaluator output is missing score(s): "
            + ", ".join(sorted(missing))
        )
    return scores["execution"], scores["exact_match"]


def evaluate_experiment(
    experiment: str,
    result_path: Path,
    dev_examples: List[Dict[str, object]],
    evaluation_dir: Path,
    database_root: Path,
    tables_path: Path,
) -> Tuple[str, float, float]:
    if not result_path.is_file():
        raise FileNotFoundError(f"Completed experiment log not found: {result_path}")
    with result_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if payload.get("experiment") != experiment:
        raise ValueError(
            f"Expected experiment {experiment!r} in {result_path}, "
            f"found {payload.get('experiment')!r}."
        )
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"Experiment records are missing in {result_path}.")

    gold_path, prediction_path = write_evaluation_inputs(
        experiment, dev_examples, records, evaluation_dir
    )
    command = [
        sys.executable,
        str(PROJECT_ROOT / "evaluation.py"),
        "--gold",
        str(gold_path),
        "--pred",
        str(prediction_path),
        "--db",
        str(database_root),
        "--table",
        str(tables_path),
        "--etype",
        "all",
    ]
    completed = subprocess.run(
        command, cwd=PROJECT_ROOT, text=True, capture_output=True
    )
    if completed.returncode:
        raise RuntimeError(
            f"Spider evaluation failed for {experiment}:\n"
            f"{completed.stderr[-4000:]}"
        )
    execution, exact_match = parse_spider_scores(completed.stdout)
    return experiment, execution, exact_match


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=RESULTS_ROOT,
        help="Root containing the experiment JSON logs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_ROOT / "dev_table_evaluation",
    )
    args = parser.parse_args()

    dev_path = PROJECT_ROOT / "data" / "spider_dev.json"
    tables_path = PROJECT_ROOT / "data" / "tables.json"
    database_root = PROJECT_ROOT / "data" / "database"
    for required in (dev_path, tables_path):
        if not required.is_file():
            raise FileNotFoundError(f"Required evaluation input not found: {required}")

    with dev_path.open("r", encoding="utf-8") as file:
        dev_examples = json.load(file)
    if not isinstance(dev_examples, list) or len(dev_examples) != 1034:
        count = len(dev_examples) if isinstance(dev_examples, list) else "invalid"
        raise ValueError(f"Expected 1,034 Spider Dev examples, found {count}.")
    dev_ids = {example["db_id"] for example in dev_examples}
    missing_databases = sorted(
        db_id
        for db_id in dev_ids
        if not (database_root / db_id / f"{db_id}.sqlite").is_file()
    )
    if missing_databases:
        raise FileNotFoundError(
            "Spider Dev databases are missing; run "
            "python data/prepare_dev_databases.py. Missing: "
            + ", ".join(missing_databases)
        )

    evaluation_dir = args.output_dir
    rows = [
        evaluate_experiment(
            experiment,
            args.results_root / EXPERIMENT_LOGS[experiment],
            dev_examples,
            evaluation_dir,
            database_root,
            tables_path,
        )
        for experiment in PAPER_EXPERIMENT_ORDER
    ]
    summary_path = evaluation_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ("experiment", "examples", "execution_accuracy", "exact_match_accuracy")
        )
        writer.writerows(
            (name, len(dev_examples), f"{execution * 100:.2f}%", f"{exact * 100:.1f}%")
            for name, execution, exact in rows
        )

    print(f"Official Spider Dev metrics ({len(dev_examples)} examples)")
    for name, execution, exact in rows:
        print(f"{name}: EX={execution * 100:.2f}% EM={exact * 100:.1f}%")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
