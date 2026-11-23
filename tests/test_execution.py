import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.execution import (
    compare_query_result_rows,
    execute_readonly_query,
    query_has_order_by,
)
from scripts.run_missing_experiments import (
    EXPERIMENTS,
    EXPERIMENT_GROUPS,
    ModelRunner,
    tokenize_generation_prompt,
)
from scripts.train_lora_variant import (
    collate_training_batch,
    encode_example,
    warmup_arguments,
)


class ReadOnlyExecutionTests(unittest.TestCase):
    def test_select_works_and_writes_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="lora test ") as directory:
            database_path = Path(directory) / "sample.sqlite"
            connection = sqlite3.connect(str(database_path))
            try:
                connection.execute("CREATE TABLE sample (value INTEGER)")
                connection.executemany(
                    "INSERT INTO sample VALUES (?)", [(1,), (1,), (2,)]
                )
                connection.commit()
            finally:
                connection.close()

            success, rows = execute_readonly_query(
                database_path, "SELECT value FROM sample ORDER BY value"
            )
            self.assertTrue(success)
            self.assertEqual(rows, [(1,), (1,), (2,)])

            success, error = execute_readonly_query(
                database_path, "DELETE FROM sample"
            )
            self.assertFalse(success)
            self.assertIn("readonly", str(error).lower())

            success, rows_after = execute_readonly_query(
                database_path, "SELECT value FROM sample ORDER BY value"
            )
            self.assertTrue(success)
            self.assertEqual(rows_after, rows)

    def test_invalid_utf8_text_is_returned_as_bytes_for_exact_comparison(self):
        with tempfile.TemporaryDirectory(prefix="lora invalid utf8 ") as directory:
            database_path = Path(directory) / "invalid.sqlite"
            connection = sqlite3.connect(str(database_path))
            try:
                connection.execute("CREATE TABLE sample (value TEXT)")
                connection.execute(
                    "INSERT INTO sample VALUES "
                    "(CAST(X'54726579657320416C62617272616EFF' AS TEXT))"
                )
                connection.commit()
            finally:
                connection.close()

            success, rows = execute_readonly_query(
                database_path, "SELECT value FROM sample"
            )
            self.assertTrue(success)
            self.assertEqual(rows, [(b"Treyes Albarran\xff",)])
            self.assertTrue(compare_query_result_rows(rows, rows))


class ExecutionAccuracyTests(unittest.TestCase):
    def test_row_order_is_ignored_but_duplicates_and_columns_are_preserved(self):
        self.assertTrue(
            compare_query_result_rows(
                [(1, "a"), (1, "a"), (2, "b")],
                [(2, "b"), (1, "a"), (1, "a")],
            )
        )
        self.assertFalse(
            compare_query_result_rows([(1, "a"), (2, "b")], [(1, "a")])
        )
        self.assertFalse(
            compare_query_result_rows([(1, "a")], [("a", 1)])
        )

    def test_null_values_compare_consistently(self):
        self.assertTrue(
            compare_query_result_rows([(None,), (1,)], [(1,), (float("nan"),)])
        )

    def test_order_by_requires_ordered_result_comparison(self):
        self.assertFalse(
            compare_query_result_rows(
                [(1,), (2,)], [(2,), (1,)], order_sensitive=True
            )
        )
        self.assertTrue(
            compare_query_result_rows(
                [(1,), (2,)], [(2,), (1,)], order_sensitive=False
            )
        )
        self.assertTrue(query_has_order_by("SELECT id FROM t ORDER BY id"))
        self.assertFalse(
            query_has_order_by("SELECT 'ORDER BY' AS note FROM t -- ORDER BY x")
        )
        self.assertFalse(
            query_has_order_by("SELECT `order by` FROM t /* ORDER BY x */")
        )


class ExperimentMatrixTests(unittest.TestCase):
    def test_matrix_maps_every_missing_table_cell_to_its_method(self):
        experiments = {
            experiment.experiment_id: (
                experiment.dataset,
                experiment.generator,
                experiment.repair,
            )
            for experiment in EXPERIMENTS
        }
        self.assertEqual(len(experiments), 17)
        self.assertEqual(
            experiments,
            {
                "zero_shot_realistic": ("spider_realistic.json", "zero_shot", "none"),
                "zero_shot_naive_repair_realistic": (
                    "spider_realistic.json", "zero_shot", "naive"
                ),
                "zero_shot_cot_realistic": (
                    "spider_realistic.json", "zero_shot_cot", "none"
                ),
                "zero_shot_cot_dev": ("spider_dev.json", "zero_shot_cot", "none"),
                "zero_shot_cot_repair_dev": (
                    "spider_dev.json", "zero_shot_cot", "cot"
                ),
                "zero_shot_cot_repair_realistic": (
                    "spider_realistic.json", "zero_shot_cot", "cot"
                ),
                "dynamic_icl_cot_dev": (
                    "spider_dev.json", "dynamic_icl_cot", "none"
                ),
                "dynamic_icl_cot_repair_dev": (
                    "spider_dev.json", "dynamic_icl_cot", "cot"
                ),
                "lora_sql_realistic": ("spider_realistic.json", "lora_sql", "none"),
                "lora_sql_naive_repair_dev": (
                    "spider_dev.json", "lora_sql", "naive"
                ),
                "lora_sql_naive_repair_realistic": (
                    "spider_realistic.json", "lora_sql", "naive"
                ),
                "lora_sql_cot_repair_dev": (
                    "spider_dev.json", "lora_sql", "cot"
                ),
                "lora_sql_cot_repair_realistic": (
                    "spider_realistic.json", "lora_sql", "cot"
                ),
                "repair_format_sft_dev": ("spider_dev.json", "zero_shot", "trained"),
                "repair_format_sft_realistic": (
                    "spider_realistic.json", "zero_shot", "trained"
                ),
                "lora_cot_naive_repair_dev": (
                    "spider_dev.json", "lora_cot", "naive"
                ),
                "lora_cot_naive_repair_realistic": (
                    "spider_realistic.json", "lora_cot", "naive"
                ),
            },
        )

    def test_lora_naive_repair_group_contains_only_missing_experiments(self):
        self.assertEqual(
            EXPERIMENT_GROUPS["lora_naive_repair"],
            (
                "lora_cot_naive_repair_dev",
                "lora_cot_naive_repair_realistic",
                "lora_sql_naive_repair_realistic",
            ),
        )


class TrainingDataTests(unittest.TestCase):
    def test_training_labels_include_prompt_and_target_tokens(self):
        class Tokenizer:
            chat_template = None
            eos_token_id = 99

            def __call__(self, text, **kwargs):
                return {"input_ids": [ord(character) for character in text]}

        encoded = encode_example(
            Tokenizer(),
            {"input_text": "prompt", "target_text": "answer"},
            max_length=32,
        )
        self.assertEqual(encoded["labels"], encoded["input_ids"])
        self.assertEqual(encoded["attention_mask"], [1] * len(encoded["input_ids"]))

    def test_chat_template_is_rendered_before_standard_tokenization(self):
        class Tokenizer:
            chat_template = "test-template"
            eos_token_id = 99

            def apply_chat_template(self, messages, tokenize, **kwargs):
                self.assert_tokenize = tokenize
                return "<user>prompt</user><assistant>SQL</assistant>"

            def __call__(self, text, **kwargs):
                assert isinstance(text, str)
                assert kwargs["add_special_tokens"] is False
                return {"input_ids": [ord(character) for character in text]}

        tokenizer = Tokenizer()
        encoded = encode_example(
            tokenizer,
            {"input_text": "prompt", "target_text": "SQL"},
            max_length=64,
        )
        self.assertFalse(tokenizer.assert_tokenize)
        self.assertTrue(all(isinstance(token_id, int) for token_id in encoded["input_ids"]))
        self.assertGreater(len(encoded["input_ids"]), len("SQL"))
        self.assertEqual(encoded["labels"], encoded["input_ids"])

    def test_empty_training_target_reports_example_context(self):
        class Tokenizer:
            chat_template = None
            eos_token_id = 99

            def __call__(self, text, **kwargs):
                return {"input_ids": [ord(character) for character in text]}

        with self.assertRaisesRegex(ValueError, "db_id=test_db"):
            encode_example(
                Tokenizer(),
                {
                    "db_id": "test_db",
                    "question": "test question",
                    "input_text": "prompt",
                    "target_text": "  ",
                },
                max_length=32,
            )

    def test_warmup_ratio_is_used_when_supported(self):
        class RatioArguments:
            def __init__(self, warmup_ratio=0.0):
                pass

        self.assertEqual(
            warmup_arguments(
                RatioArguments, {"warmup_ratio": 0.03, "max_steps": 200}
            ),
            {"warmup_ratio": 0.03},
        )

    def test_warmup_ratio_converts_to_steps_for_older_api(self):
        class StepsArguments:
            def __init__(self, warmup_steps=0):
                pass

        self.assertEqual(
            warmup_arguments(
                StepsArguments, {"warmup_ratio": 0.03, "max_steps": 200}
            ),
            {"warmup_steps": 6},
        )

    def test_batch_collator_pads_token_lists_without_tokenizer_pad(self):
        import torch

        batch = collate_training_batch(
            [
                {"input_ids": [10, 11], "labels": [10, 11]},
                {"input_ids": [20], "labels": [20]},
            ],
            pad_token_id=0,
        )
        self.assertTrue(torch.equal(
            batch["input_ids"], torch.tensor([[10, 11], [20, 0]])
        ))
        self.assertTrue(torch.equal(
            batch["attention_mask"], torch.tensor([[1, 1], [1, 0]])
        ))
        self.assertTrue(torch.equal(
            batch["labels"], torch.tensor([[10, 11], [20, -100]])
        ))

    def test_batch_collator_reports_string_token_ids(self):
        with self.assertRaisesRegex(TypeError, "invalid input_ids"):
            collate_training_batch(
                [{"input_ids": "not token IDs", "labels": [1]}],
                pad_token_id=0,
            )

    def test_batch_collator_normalizes_nested_and_tensor_token_ids(self):
        import torch

        batch = collate_training_batch(
            [
                {
                    "input_ids": torch.tensor([[10, 11]]),
                    "labels": torch.tensor([[10, 11]]),
                },
                {
                    "input_ids": [[20]],
                    "labels": [[20]],
                },
            ],
            pad_token_id=0,
        )
        self.assertTrue(torch.equal(
            batch["input_ids"], torch.tensor([[10, 11], [20, 0]])
        ))
        self.assertTrue(torch.equal(
            batch["labels"], torch.tensor([[10, 11], [20, -100]])
        ))

    def test_batch_collator_rejects_non_integer_token_tensor(self):
        import torch

        with self.assertRaisesRegex(TypeError, "non-integer input_ids dtype"):
            collate_training_batch(
                [{"input_ids": torch.tensor([1.0, 2.0]), "labels": [1, 2]}],
                pad_token_id=0,
            )


class InferenceTokenizationTests(unittest.TestCase):
    def make_runner(self, tokenizer):
        import torch

        runner = ModelRunner.__new__(ModelRunner)
        runner.torch = torch
        runner.tokenizer = tokenizer
        runner.max_input_tokens = 8
        runner.generation_config = {
            "do_sample": False,
        }
        runner.adapter_names = {}

        class Model:
            device = torch.device("cpu")

            def eval(self):
                return self

            def generate(self, input_ids, **kwargs):
                self.received_shape = tuple(input_ids.shape)
                return torch.cat(
                    [input_ids, torch.tensor([[42]], device=input_ids.device)],
                    dim=1,
                )

        runner.model = Model()
        return runner

    def test_chat_prompt_is_rendered_then_tokenized_normally(self):
        import torch

        class Tokenizer:
            chat_template = "test-template"
            pad_token_id = 0
            eos_token_id = 99

            def apply_chat_template(self, messages, tokenize, **kwargs):
                assert tokenize is False
                return "<|user|>hello<|assistant|>"

            def __call__(self, text, **kwargs):
                assert text == "<|user|>hello<|assistant|>"
                return {"input_ids": torch.tensor([[1, 2, 3]])}

            def decode(self, tokens, skip_special_tokens):
                return str(tokens.tolist())

        runner = self.make_runner(Tokenizer())
        self.assertEqual(runner.generate("hello", None, 1), "[42]")
        self.assertEqual(runner.model.received_shape, (1, 3))

    def test_chat_template_returning_string_ids_is_not_used(self):
        import torch

        class Tokenizer:
            chat_template = "test-template"
            pad_token_id = 0
            eos_token_id = 99

            def apply_chat_template(self, messages, tokenize, **kwargs):
                assert tokenize is False
                return "<rendered chat>"

            def __call__(self, text, **kwargs):
                return {"input_ids": torch.tensor([[5, 6, 7]])}

            def decode(self, tokens, skip_special_tokens):
                return str(tokens.tolist())

        runner = self.make_runner(Tokenizer())
        self.assertEqual(runner.generate("hello", None, 1), "[42]")
        self.assertEqual(runner.model.received_shape, (1, 3))

    def test_non_chat_prompt_is_tokenized_into_tensor(self):
        import torch

        class Tokenizer:
            chat_template = None
            pad_token_id = 0
            eos_token_id = 99

            def __call__(self, text, **kwargs):
                return {"input_ids": torch.tensor([[8, 9]])}

            def decode(self, tokens, skip_special_tokens):
                return str(tokens.tolist())

        runner = self.make_runner(Tokenizer())
        self.assertEqual(runner.generate("hello", None, 1), "[42]")
        self.assertEqual(runner.model.received_shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
