"""Run the experiment cells that are still TBD in the comparison table."""

import argparse
import csv
import gc
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.execution import (
    clean_sql_from_response,
    compare_query_result_rows,
    execute_readonly_query,
    query_has_order_by,
)
from src.prompts import (
    prompt_generator_cot,
    prompt_m1_zeroshot,
    prompt_naive_repair,
    prompt_repair_cot,
)
from src.schema_engine import extract_schema_ddl, locate_database_path

DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "artifacts" / "results_json" / "missing_experiments"
)
DYNAMIC_ICL_POOL_SIZE = 1500


def tokenize_generation_prompt(tokenizer, prompt: str, torch_module):
    if getattr(tokenizer, "chat_template", None):
        rendered_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(rendered_prompt, str):
            raise TypeError(
                "Tokenizer chat template must return rendered text when "
                "tokenize=False"
            )
    else:
        rendered_prompt = prompt

    encoded = tokenizer(
        rendered_prompt, return_tensors="pt", truncation=False
    )
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if not isinstance(input_ids, torch_module.Tensor):
        try:
            input_ids = torch_module.as_tensor(input_ids, dtype=torch_module.long)
        except (TypeError, ValueError, RuntimeError) as error:
            raise TypeError(
                "Tokenizer must return numeric input_ids after tokenizing "
                f"the rendered prompt; got {type(input_ids).__name__}"
            ) from error
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(
            "Prompt tokenization must produce input_ids with shape "
            f"(1, sequence_length); got {tuple(input_ids.shape)}"
        )
    return input_ids


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    dataset: str
    generator: str
    repair: str


EXPERIMENTS = (
    Experiment("zero_shot_realistic", "spider_realistic.json", "zero_shot", "none"),
    Experiment("zero_shot_naive_repair_realistic", "spider_realistic.json", "zero_shot", "naive"),
    Experiment("zero_shot_cot_realistic", "spider_realistic.json", "zero_shot_cot", "none"),
    Experiment("zero_shot_cot_dev", "spider_dev.json", "zero_shot_cot", "none"),
    Experiment("zero_shot_cot_repair_dev", "spider_dev.json", "zero_shot_cot", "cot"),
    Experiment("zero_shot_cot_repair_realistic", "spider_realistic.json", "zero_shot_cot", "cot"),
    Experiment("dynamic_icl_cot_dev", "spider_dev.json", "dynamic_icl_cot", "none"),
    Experiment("dynamic_icl_cot_repair_dev", "spider_dev.json", "dynamic_icl_cot", "cot"),
    Experiment("lora_sql_realistic", "spider_realistic.json", "lora_sql", "none"),
    Experiment("lora_sql_naive_repair_dev", "spider_dev.json", "lora_sql", "naive"),
    Experiment("lora_sql_naive_repair_realistic", "spider_realistic.json", "lora_sql", "naive"),
    Experiment("lora_sql_cot_repair_dev", "spider_dev.json", "lora_sql", "cot"),
    Experiment("lora_sql_cot_repair_realistic", "spider_realistic.json", "lora_sql", "cot"),
    Experiment("repair_format_sft_dev", "spider_dev.json", "zero_shot", "trained"),
    Experiment("repair_format_sft_realistic", "spider_realistic.json", "zero_shot", "trained"),
    Experiment("lora_cot_naive_repair_dev", "spider_dev.json", "lora_cot", "naive"),
    Experiment("lora_cot_naive_repair_realistic", "spider_realistic.json", "lora_cot", "naive"),
)
EXPERIMENT_BY_ID = {experiment.experiment_id: experiment for experiment in EXPERIMENTS}
EXPERIMENT_GROUPS = {
    "lora_naive_repair": (
        "lora_cot_naive_repair_dev",
        "lora_cot_naive_repair_realistic",
        "lora_sql_naive_repair_realistic",
    ),
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    temporary_path.replace(path)


def format_rows_equal(
    gold_rows: List[Tuple[object, ...]],
    predicted_rows: List[Tuple[object, ...]],
    order_sensitive: bool = False,
) -> bool:
    """Compatibility wrapper for the shared execution-result comparator."""
    return compare_query_result_rows(
        gold_rows, predicted_rows, order_sensitive=order_sensitive
    )


def generation_status(
    sql: str,
    gold_rows: List[Tuple[object, ...]],
    db_path: Path,
    timeout_seconds: float,
    order_sensitive: bool,
) -> Tuple[str, Optional[str]]:
    success, result = execute_readonly_query(db_path, sql, timeout_seconds)
    if not success:
        return "SYNTAX_ERROR", str(result)
    return (
        "CORRECT" if format_rows_equal(
            gold_rows, result, order_sensitive=order_sensitive
        ) else "SEMANTIC_ERROR",
        None,
    )


class ModelRunner:
    def __init__(
        self,
        model_id: str,
        generator_adapter: Optional[Path],
        repair_adapter: Optional[Path],
        max_input_tokens: int,
        generation_config: Dict[str, object],
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Inference requires torch and transformers from requirements.txt."
            ) from error

        self.torch = torch
        self.max_input_tokens = max_input_tokens
        self.generation_config = generation_config
        adapters = [
            ("generator", generator_adapter),
            ("repair", repair_adapter),
        ]
        adapters = [(name, path) for name, path in adapters if path is not None]
        for _, path in adapters:
            self._check_adapter(path)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {"trust_remote_code": True}
        if torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        self.adapter_names: Dict[str, str] = {}

        if adapters:
            try:
                from peft import PeftModel
            except ImportError as error:
                raise RuntimeError(
                    "LoRA inference requires peft. Install requirements.txt."
                ) from error

            first_name, first_path = adapters[0]
            self._check_adapter(first_path)
            model = PeftModel.from_pretrained(
                model, str(first_path), adapter_name=first_name
            )
            self.adapter_names[first_name] = first_name
            for name, path in adapters[1:]:
                if path.resolve() == first_path.resolve():
                    self.adapter_names[name] = first_name
                    continue
                self._check_adapter(path)
                model.load_adapter(str(path), adapter_name=name)
                self.adapter_names[name] = name

        self.model = model
        self.model.eval()

    @staticmethod
    def _check_adapter(path: Path) -> None:
        if not path.is_dir() or not (path / "adapter_config.json").is_file():
            raise FileNotFoundError(
                f"LoRA adapter is not trained or is incomplete: {path}. "
                "Prepare data and run scripts/train_lora_variant.py first."
            )
        weight_files = ("adapter_model.safetensors", "adapter_model.bin")
        if not any((path / weight_name).is_file() for weight_name in weight_files):
            raise FileNotFoundError(f"Adapter weights are missing from {path}")

    def generate(self, prompt: str, adapter: Optional[str], max_new_tokens: int) -> str:
        if adapter is not None:
            if adapter not in self.adapter_names:
                raise ValueError(f"Requested adapter was not loaded: {adapter}")
            self.model.set_adapter(self.adapter_names[adapter])

        input_ids = tokenize_generation_prompt(
            self.tokenizer, prompt, self.torch
        )
        input_ids = input_ids[:, -self.max_input_tokens:]
        input_ids = input_ids.to(self.model.device)

        def generate_tokens():
            with self.torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=bool(self.generation_config["do_sample"]),
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            return output_ids[0, input_ids.shape[1]:]

        if adapter is None and self.adapter_names:
            with self.model.disable_adapter():
                new_tokens = generate_tokens()
        else:
            new_tokens = generate_tokens()
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


def load_retriever(data_dir: Path, db_root: Path):
    train_path = data_dir / "train_spider.json"
    dev_path = data_dir / "spider_dev.json"
    realistic_path = data_dir / "spider_realistic.json"
    missing = [path for path in (train_path, dev_path, realistic_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Dynamic ICL requires train_spider.json, spider_dev.json, and "
            "spider_realistic.json. Run data/download_and_prepare.py first."
        )

    excluded_db_ids = {
        sample["db_id"]
        for path in (dev_path, realistic_path)
        for sample in load_json(path)
    }
    train_pool = [
        sample for sample in load_json(train_path)
        if sample["db_id"] not in excluded_db_ids
    ][:DYNAMIC_ICL_POOL_SIZE]
    if not train_pool:
        raise RuntimeError("No leakage-free Spider training examples for Dynamic ICL")
    try:
        from src.retriever import DynamicICLRetriever
    except ImportError as error:
        raise RuntimeError(
            "Dynamic ICL requires sentence-transformers; install requirements.txt."
        ) from error
    return DynamicICLRetriever(train_pool, db_root_dir=db_root)


def load_state(
    result_path: Path,
    experiment: Experiment,
    config: Dict[str, object],
    overwrite: bool,
) -> Dict[int, Dict[str, object]]:
    if overwrite or not result_path.exists():
        return {}
    payload = load_json(result_path)
    if payload.get("experiment") != experiment.experiment_id or payload.get("config") != config:
        raise ValueError(
            f"{result_path} belongs to a different experiment configuration. "
            "Use --overwrite to replace it."
        )
    return {int(record["index"]): record for record in payload.get("records", [])}


def update_summary(summary_path: Path, rows: List[Dict[str, object]]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ("experiment", "dataset", "examples", "correct", "execution_accuracy")
    temporary_path = summary_path.with_suffix(summary_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(summary_path)


def validate_prerequisites(experiments: Tuple[Experiment, ...], args) -> None:
    required_adapter_paths = set()
    required_databases = set()
    required_datasets = set()
    for experiment in experiments:
        dataset_path = args.data_dir / experiment.dataset
        required_datasets.add(dataset_path)
        if experiment.generator == "lora_sql":
            required_adapter_paths.add(args.adapter_sql)
        elif experiment.generator == "lora_cot":
            required_adapter_paths.add(args.adapter_cot)
        if experiment.repair == "trained":
            required_adapter_paths.add(args.adapter_repair)
        if dataset_path.is_file():
            examples = load_json(dataset_path)
            if args.limit is not None:
                examples = examples[:args.limit]
            required_databases.update(sample["db_id"] for sample in examples)

    missing_datasets = [path for path in required_datasets if not path.is_file()]
    if missing_datasets:
        raise FileNotFoundError(
            "Missing evaluation data: "
            + ", ".join(str(path) for path in missing_datasets)
            + ". Run data/download_and_prepare.py first."
        )
    for adapter_path in required_adapter_paths:
        ModelRunner._check_adapter(adapter_path)
    if any(item.generator == "dynamic_icl_cot" for item in experiments):
        if not (args.data_dir / "train_spider.json").is_file():
            raise FileNotFoundError(
                "Dynamic ICL requires train_spider.json. "
                "Run data/download_and_prepare.py first."
            )
    missing_db_ids = [
        db_id for db_id in sorted(required_databases)
        if not locate_database_path(db_id, args.db_root).is_file()
    ]
    if missing_db_ids:
        raise FileNotFoundError(
            "Missing Spider SQLite databases: "
            + ", ".join(missing_db_ids[:10])
            + (" ..." if len(missing_db_ids) > 10 else "")
            + ". Run data/download_and_prepare.py first."
        )


def run_experiment(
    experiment: Experiment,
    args,
    model_id: str,
    generation_config: Dict[str, object],
    gold_cache: Dict[Tuple[str, str], List[Tuple[object, ...]]],
    retriever,
) -> Dict[str, object]:
    dataset_path = args.data_dir / experiment.dataset
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    examples = load_json(dataset_path)
    if args.limit is not None:
        examples = examples[:args.limit]

    adapter_paths = {
        "lora_sql": args.adapter_sql,
        "lora_cot": args.adapter_cot,
    }
    generator_adapter = adapter_paths.get(experiment.generator)
    repair_adapter = args.adapter_repair if experiment.repair == "trained" else None
    config = {
        "model_id": model_id,
        "generator": experiment.generator,
        "repair": experiment.repair,
        "dataset_path": str(dataset_path.resolve()),
        "generator_adapter": str(generator_adapter) if generator_adapter else None,
        "repair_adapter": str(repair_adapter) if repair_adapter else None,
        "limit": args.limit,
        "top_k": args.top_k,
        "max_input_tokens": args.max_input_tokens,
        "sql_timeout": args.sql_timeout,
        "generation_config": generation_config,
    }
    result_path = args.output_dir / f"{experiment.experiment_id}.json"
    records_by_index = load_state(
        result_path, experiment, config, args.overwrite
    )
    pending = [
        (index, example)
        for index, example in enumerate(examples)
        if index not in records_by_index
    ]

    if pending:
        runner = ModelRunner(
            model_id,
            generator_adapter,
            repair_adapter,
            args.max_input_tokens,
            generation_config,
        )
        schema_cache: Dict[str, str] = {}
        for index, example in pending:
            db_id = example["db_id"]
            db_path = locate_database_path(db_id, args.db_root)
            if not db_path.is_file():
                raise FileNotFoundError(
                    f"SQLite database for {db_id} was not found at {db_path}. "
                    "Run data/download_and_prepare.py first."
                )
            schema = schema_cache.get(db_id)
            if schema is None:
                schema = extract_schema_ddl(db_path)
                if not schema:
                    raise RuntimeError(f"No schema found in database {db_path}")
                schema_cache[db_id] = schema

            gold_sql = example.get("query") or example.get("sql")
            if not gold_sql:
                raise ValueError(f"Spider example {index} has no gold SQL query")
            gold_key = (db_id, gold_sql)
            if gold_key not in gold_cache:
                gold_ok, gold_result = execute_readonly_query(
                    db_path, gold_sql, args.sql_timeout
                )
                if not gold_ok:
                    raise RuntimeError(
                        f"Gold SQL failed for {experiment.experiment_id}, "
                        f"record {index}, database {db_id}: {gold_result}"
                    )
                gold_cache[gold_key] = gold_result
            gold_rows = gold_cache[gold_key]

            question = example["question"]
            if experiment.generator == "dynamic_icl_cot":
                prompt = retriever.build_dynamic_prompt(
                    schema, question, top_k=args.top_k, with_cot=True
                )
            elif experiment.generator in ("zero_shot_cot", "lora_cot"):
                prompt = prompt_generator_cot(schema, question)
            else:
                prompt = prompt_m1_zeroshot(schema, question)

            adapter = "generator" if generator_adapter else None
            raw_initial = runner.generate(
                prompt,
                adapter,
                int(generation_config["max_new_tokens_generator"]),
            )
            initial_sql = clean_sql_from_response(raw_initial)
            pre_status, initial_error = generation_status(
                initial_sql, gold_rows, db_path, args.sql_timeout,
                query_has_order_by(gold_sql),
            )

            final_sql = initial_sql
            final_status = pre_status
            repair_sql = None
            repair_response = None
            repair_attempted = False
            if pre_status == "SYNTAX_ERROR" and experiment.repair != "none":
                repair_attempted = True
                if experiment.repair == "cot":
                    repair_prompt = prompt_repair_cot(
                        schema, question, initial_sql, initial_error or ""
                    )
                else:
                    repair_prompt = prompt_naive_repair(
                        schema, question, initial_sql, initial_error or ""
                    )
                repair_response = runner.generate(
                    repair_prompt,
                    "repair" if experiment.repair == "trained" else adapter,
                    int(generation_config["max_new_tokens_repair"]),
                )
                repair_sql = clean_sql_from_response(repair_response)
                final_sql = repair_sql
                final_status, _ = generation_status(
                    repair_sql, gold_rows, db_path, args.sql_timeout,
                    query_has_order_by(gold_sql),
                )

            records_by_index[index] = {
                "index": index,
                "db_id": db_id,
                "difficulty": example.get("difficulty", ""),
                "question": question,
                "pre_status": pre_status,
                "post_status": final_status,
                "initial_sql": initial_sql,
                "repair_sql": repair_sql,
                "repair_attempted": repair_attempted,
            }
            write_json_atomic(result_path, {
                "experiment": experiment.experiment_id,
                "dataset": experiment.dataset,
                "config": config,
                "records": [
                    records_by_index[key] for key in sorted(records_by_index)
                ],
            })

        del runner
        gc.collect()
        if args.clear_cuda_cache:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    records = [records_by_index[index] for index in sorted(records_by_index)]
    correct = sum(record["post_status"] == "CORRECT" for record in records)
    total = len(records)
    row = {
        "experiment": experiment.experiment_id,
        "dataset": experiment.dataset,
        "examples": total,
        "correct": correct,
        "execution_accuracy": f"{100 * correct / total:.2f}%" if total else "N/A",
    }
    print(
        f"{experiment.experiment_id}: "
        f"{row['execution_accuracy']} ({correct}/{total})"
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=("all",) + tuple(EXPERIMENT_GROUPS) + tuple(EXPERIMENT_BY_ID),
        default="all",
    )
    parser.add_argument("--list", action="store_true", help="List experiment IDs and exit")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--db-root", type=Path, default=DATA_DIR / "database")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-id", default=None)
    parser.add_argument(
        "--adapter-sql",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "checkpoints" / "lora_sql_adapter",
    )
    parser.add_argument(
        "--adapter-cot",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "checkpoints" / "lora_cot_adapter",
    )
    parser.add_argument(
        "--adapter-repair",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "checkpoints" / "lora_repair_format_adapter",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--sql-timeout", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--clear-cuda-cache", action="store_true")
    args = parser.parse_args()
    if args.list:
        for experiment in EXPERIMENTS:
            print(
                f"{experiment.experiment_id}\t{experiment.dataset}\t"
                f"{experiment.generator}\t{experiment.repair}"
            )
        return
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.top_k < 1 or args.max_input_tokens < 1 or args.sql_timeout <= 0:
        parser.error("--top-k, --max-input-tokens, and --sql-timeout must be positive")

    with (PROJECT_ROOT / "configs" / "train_config.json").open(
        "r", encoding="utf-8"
    ) as file:
        train_config = json.load(file)
    with (PROJECT_ROOT / "configs" / "generation_config.json").open(
        "r", encoding="utf-8"
    ) as file:
        generation_config = json.load(file)
    model_id = args.model_id or train_config["model_id"]

    if args.experiment == "all":
        selected = EXPERIMENTS
    elif args.experiment in EXPERIMENT_GROUPS:
        selected = tuple(
            EXPERIMENT_BY_ID[experiment_id]
            for experiment_id in EXPERIMENT_GROUPS[args.experiment]
        )
    else:
        selected = (EXPERIMENT_BY_ID[args.experiment],)
    validate_prerequisites(selected, args)
    retriever = None
    if any(item.generator == "dynamic_icl_cot" for item in selected):
        retriever = load_retriever(args.data_dir, args.db_root)

    summary_path = args.output_dir / "summary.csv"
    summary_rows: List[Dict[str, object]] = []
    if summary_path.exists() and not args.overwrite:
        with summary_path.open("r", encoding="utf-8", newline="") as file:
            summary_rows = list(csv.DictReader(file))
    summary_by_id = {
        str(row["experiment"]): row
        for row in summary_rows
        if str(row["experiment"]) in EXPERIMENT_BY_ID
    }
    gold_cache: Dict[Tuple[str, str], List[Tuple[object, ...]]] = {}
    for experiment in selected:
        row = run_experiment(
            experiment,
            args,
            model_id,
            generation_config,
            gold_cache,
            retriever,
        )
        summary_by_id[experiment.experiment_id] = row
        order = {item.experiment_id: index for index, item in enumerate(EXPERIMENTS)}
        ordered_rows = sorted(
            summary_by_id.values(),
            key=lambda item: order.get(str(item["experiment"]), len(order)),
        )
        update_summary(summary_path, ordered_rows)


if __name__ == "__main__":
    main()
