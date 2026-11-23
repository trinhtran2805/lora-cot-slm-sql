# Data

This release includes the raw Spider train examples, the 1,034-question Spider
Dev split, its table metadata and compressed SQLite databases, and the
preprocessed 800-example SQL, CoT, and repair training sets.

Extract the Dev databases from the bundled archive:

```bash
python data/prepare_dev_databases.py
```

`spider_realistic.json` is included only so `prepare_experiment_data.py` can
exclude its database IDs from training. No Spider-Realistic evaluation results
are included in this repository.

To regenerate the preprocessed training sets from raw Spider data, the full
Spider SQLite database collection is needed. It is not bundled because of its
size and GitHub's per-file limit. Download it from the upstream source with:

```bash
python data/download_and_prepare.py --full-spider
python data/prepare_experiment_data.py --format sql
python data/prepare_experiment_data.py --format cot
python data/prepare_experiment_data.py --format repair
```

The downloader also obtains Spider-Realistic for leakage filtering. Follow the
upstream dataset terms and cite Spider as specified by its authors.
