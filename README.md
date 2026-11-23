# LoRA-CoT-Repair

Code and data for reproducing the **Spider Dev** results in the paper
“Coupling Parameter-Efficient Fine-Tuning with Reasoning-Guided Self-Correction
for Small Language Models in Text-to-SQL Semantic Parsing.”

This snapshot includes the Spider train and Dev examples, the preprocessed
training sets used by the notebooks, and a compressed bundle of the 20 SQLite
databases used by the 1,034-example Dev split. It does not include model
checkpoints or run logs; train the adapters and generate those locally.

## What is included

- `data/train_spider.json`, `data/spider_dev.json`, and `data/tables.json`.
- `data/spider_dev_databases.zip` and a script to extract the Dev SQLite files.
- The three 800-example SQL, CoT, and repair training sets used by the training
  script.
- `src/`, `scripts/`, `configs/`, and focused tests.
- `results/paper_main_table.csv`, with the paper table's row order and component
  markers and EX/EM recomputed consistently using the official Spider evaluator.

`data/spider_realistic.json` is retained only because the training-data
preparation script excludes its database IDs to prevent evaluation leakage.
No Spider-Realistic results are part of this Dev results table.

The full Spider database collection is not checked in: it is large and has
files exceeding GitHub's per-file limit. The bundled Dev database archive is
39.6 MB; the training inputs themselves are checked in, so it is not necessary
to download the full collection to train the paper adapters.

## Requirements and setup

- Python 3.10 or later
- PyTorch with CUDA recommended for adapter training and full evaluation
- Access to `Qwen/Qwen2.5-Coder-1.5B-Instruct` from Hugging Face

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

Extract the bundled Dev databases:

```bash
python data/prepare_dev_databases.py
```

The script validates that the archive contains exactly the databases referenced
by `spider_dev.json`. The extracted `data/database/` directory is ignored by
Git.

## Train the adapters

The prepared training data is included, so these commands do not require the
full Spider SQLite database collection:

```bash
python scripts/train_lora_variant.py --format sql
python scripts/train_lora_variant.py --format cot
```

The main table also includes zero-shot and repair comparisons. To regenerate
the prepared training files from the raw training examples and SQLite
databases, download the full Spider archive:

```bash
python data/download_and_prepare.py --full-spider
python data/prepare_experiment_data.py --format sql
python data/prepare_experiment_data.py --format cot
python data/prepare_experiment_data.py --format repair
```

This full-data preparation downloads Spider and Spider-Realistic from their
upstream sources; review their terms before using or redistributing those
datasets. The SQL-only adapter uses the sequence limit from experiment 1;
CoT and repair-format adapters use the experiment 2 limit. See `configs/` for
the training and generation settings.

## Reproduce the Dev table

Run the nine standard Dev experiments, including the added zero-shot-CoT +
naive-repair comparison, and train the separate SQL adapter used by the
LoRA-SQL + CoT-repair row:

```bash
python scripts/train_lora_variant.py --format sql --output-dir artifacts/checkpoints/lora_sql_adapter_dev_run_2
python scripts/run_missing_experiments.py --experiment lora_sql_cot_repair_dev --adapter-sql artifacts/checkpoints/lora_sql_adapter_dev_run_2 --output-dir artifacts/results_json/lora_sql_cot_repair_dev_full
python scripts/evaluate_dev_table.py
```

To run and officially score only the added zero-shot-CoT + naive-repair
configuration:

```bash
python scripts/run_missing_experiments.py --experiment zero_shot_cot_naive_repair_dev
python scripts/evaluate_dev_table.py --experiment zero_shot_cot_naive_repair_dev --output-dir artifacts/results_json/zero_shot_cot_naive_repair_dev_evaluation
```

This is a zero-shot inference condition: it uses the base model for both CoT
generation and naive repair, not an adapter trained on 800 examples.

Then compute both metrics for all ten rows with the official Spider evaluator
used in the notebooks:

```bash
python scripts/evaluate_dev_table.py
```

The resulting `artifacts/results_json/dev_table_evaluation/summary.csv`
contains execution accuracy (EX) and exact-match accuracy (EM) for all ten
rows, evaluated with `evaluation.py --etype all`. The new row is pending until
its experiment log is generated; do not treat its blank metrics as a result.
The other nine rows contain the previously established metrics.

The experiment runner also writes `summary.csv` files by counting records whose
`post_status` is `CORRECT`. That runner-level result comparator is a different
execution-scoring path from Spider's official evaluator, so those percentages
must not be mixed into the main table. For example, the runner reports 59.86%
for `lora_sql_cot_repair_dev`, while the official Spider evaluator reports
58.60% EX / 51.8% EM. This distinction applies to the other experiments too.

Use `--limit 5` for a smoke run, for example:

```bash
python scripts/run_missing_experiments.py --experiment dynamic_icl_cot_dev --limit 5
```

## Main results

`results/paper_main_table.csv` preserves the table's component markers and row
order. Its nine EX/EM pairs were recomputed from the matching per-example logs
with the official Spider evaluator. The `matched_experiment_id` column links
each metric row to the corresponding experiment log name.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Citation

```bibtex
@article{trinh2026loracotrepair,
  title={LoRA-CoT-Repair: Coupling Parameter-Efficient Fine-Tuning with
         Reasoning-Guided Self-Correction for SLM Text-to-SQL Parsing},
  author={Tran, Thi Dieu-Trinh and Le, Quang-Hung},
  year={2026}
}
```

Add the final venue, DOI, and publication metadata when available. When using
Spider, cite Yu et al. (EMNLP 2018), as requested by the upstream dataset.

## License

The project code is distributed under the MIT License. Spider data is
distributed under its upstream license; the base model and Spider-Realistic
data remain subject to their own terms.
