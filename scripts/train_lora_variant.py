"""Train one SQL, rationale, or repair-format LoRA adapter."""

import argparse
import inspect
import json
from numbers import Integral
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

VARIANTS = {
    "sql": ("sql", "lora_sql_adapter"),
    "cot": ("cot", "lora_cot_adapter"),
    "repair": ("repair", "lora_repair_format_adapter"),
}


def encode_example(tokenizer, example: Dict[str, str], max_length: int) -> Dict[str, List[int]]:
    target_text = example.get("target_text", "")
    if not target_text.strip():
        raise ValueError(
            f"Training example has an empty target_text "
            f"(db_id={example.get('db_id', 'unknown')}, "
            f"question={example.get('question', 'unknown')!r})"
        )

    messages = [{"role": "user", "content": example["input_text"]}]
    if getattr(tokenizer, "chat_template", None):
        full_text = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": target_text}],
            tokenize=False,
            add_generation_prompt=False,
        )
        if not isinstance(full_text, str):
            raise TypeError(
                "Tokenizer chat template must return rendered text when "
                "tokenize=False"
            )
        tokenized = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        if hasattr(tokenized, "tolist"):
            tokenized = tokenized.tolist()
        if (
            isinstance(tokenized, (list, tuple))
            and len(tokenized) == 1
            and isinstance(tokenized[0], (list, tuple))
        ):
            tokenized = tokenized[0]
        if not isinstance(tokenized, (list, tuple)) or not all(
            isinstance(token_id, Integral) for token_id in tokenized
        ):
            raise TypeError(
                "Tokenizer must return integer input_ids for the rendered "
                f"chat; got {type(tokenized).__name__}"
            )
        input_ids = [int(token_id) for token_id in tokenized]

        target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
        if hasattr(target_ids, "tolist"):
            target_ids = target_ids.tolist()
        if (
            isinstance(target_ids, (list, tuple))
            and len(target_ids) == 1
            and isinstance(target_ids[0], (list, tuple))
        ):
            target_ids = target_ids[0]
        if not isinstance(target_ids, (list, tuple)) or not all(
            isinstance(token_id, Integral) for token_id in target_ids
        ):
            raise TypeError("Tokenizer must return integer token IDs for target_text")
        if len(target_ids) >= max_length:
            raise ValueError(
                f"Target has {len(target_ids)} tokens, not less than "
                f"max_seq_length={max_length}"
            )
        input_ids = input_ids[-max_length:]
    else:
        prefix_ids = tokenizer(
            example["input_text"], add_special_tokens=True
        )["input_ids"]
        target_ids = tokenizer(
            target_text, add_special_tokens=False
        )["input_ids"]
        if tokenizer.eos_token_id is not None:
            target_ids = target_ids + [tokenizer.eos_token_id]

        if not target_ids:
            raise ValueError(
                "Tokenizer produced no tokens for non-empty target_text "
                f"(db_id={example.get('db_id', 'unknown')}, "
                f"question={example.get('question', 'unknown')!r})"
            )
        if len(target_ids) >= max_length:
            raise ValueError(
                f"Target has {len(target_ids)} tokens, not less than "
                f"max_seq_length={max_length}"
            )

        prefix_limit = max_length - len(target_ids)
        prefix_ids = prefix_ids[-prefix_limit:] if prefix_limit else []
        input_ids = prefix_ids + target_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": input_ids.copy(),
    }


def warmup_arguments(training_arguments_class, train_config: Dict[str, object]) -> Dict[str, object]:
    parameters = inspect.signature(training_arguments_class).parameters
    warmup_ratio = float(train_config["warmup_ratio"])
    if "warmup_ratio" in parameters:
        return {"warmup_ratio": warmup_ratio}
    if "warmup_steps" in parameters:
        return {
            "warmup_steps": int(
                int(train_config["max_steps"]) * warmup_ratio
            )
        }
    raise TypeError(
        "Installed TrainingArguments supports neither warmup_ratio nor "
        "warmup_steps; install a compatible transformers version."
    )


def collate_training_batch(features, pad_token_id: int, padding_side: str = "right"):
    import torch

    if padding_side not in ("left", "right"):
        raise ValueError(f"Unsupported tokenizer padding side: {padding_side}")
    if not features:
        raise ValueError("Cannot collate an empty training batch")

    def token_list(value, field_name: str, item_index: int):
        try:
            tensor = torch.as_tensor(value)
        except (TypeError, ValueError, RuntimeError) as error:
            raise TypeError(
                f"Training item {item_index} has invalid {field_name} "
                f"({type(value).__name__}); expected integer token IDs"
            ) from error

        while tensor.ndim > 1 and 1 in tensor.shape:
            tensor = tensor.squeeze(next(
                axis for axis, size in enumerate(tensor.shape) if size == 1
            ))
        if tensor.ndim != 1:
            raise ValueError(
                f"Training item {item_index} has {field_name} shape "
                f"{tuple(tensor.shape)}; expected a 1D token sequence"
            )
        if tensor.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise TypeError(
                f"Training item {item_index} has non-integer {field_name} "
                f"dtype {tensor.dtype}; expected integer token IDs"
            )
        return tensor.to(dtype=torch.long).tolist()

    input_sequences = []
    label_sequences = []
    for index, feature in enumerate(features):
        input_ids = token_list(feature.get("input_ids"), "input_ids", index)
        labels = token_list(feature.get("labels"), "labels", index)
        if len(labels) != len(input_ids):
            raise ValueError(
                f"Training item {index} has {len(input_ids)} input tokens "
                f"but {len(labels)} labels"
            )
        input_sequences.append(input_ids)
        label_sequences.append(labels)

    max_length = max(len(sequence) for sequence in input_sequences)
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []
    for input_ids, labels in zip(input_sequences, label_sequences):
        padding_length = max_length - len(input_ids)
        input_padding = [pad_token_id] * padding_length
        mask_padding = [0] * padding_length
        label_padding = [-100] * padding_length
        if padding_side == "right":
            batch_input_ids.append(input_ids + input_padding)
            batch_attention_mask.append([1] * len(input_ids) + mask_padding)
            batch_labels.append(labels + label_padding)
        else:
            batch_input_ids.append(input_padding + input_ids)
            batch_attention_mask.append(mask_padding + [1] * len(input_ids))
            batch_labels.append(label_padding + labels)

    return {
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        "labels": torch.tensor(batch_labels, dtype=torch.long),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as error:
        raise RuntimeError(
            "Training requires the packages in requirements.txt, including peft."
        ) from error

    format_name, adapter_name = VARIANTS[args.format]
    with (PROJECT_ROOT / "configs" / "train_config.json").open(
        "r", encoding="utf-8"
    ) as file:
        train_config = json.load(file)
    with (PROJECT_ROOT / "configs" / "lora_config.json").open(
        "r", encoding="utf-8"
    ) as file:
        lora_config = json.load(file)

    sample_count = train_config["num_train_samples"]
    data_path = args.data or (
        PROJECT_ROOT / "data" / "processed"
        / f"{format_name}_train_{sample_count}.json"
    )
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Training data not found: {data_path}. "
            f"Run data/prepare_experiment_data.py --format {args.format} first."
        )
    with data_path.open("r", encoding="utf-8") as file:
        samples = json.load(file)
    if len(samples) != sample_count:
        raise ValueError(
            f"Expected {sample_count} samples from train_config.json; found {len(samples)}"
        )

    output_dir = args.output_dir or (
        PROJECT_ROOT / "artifacts" / "checkpoints" / adapter_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        train_config["model_id"], trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded_samples = [
        encode_example(
            tokenizer,
            sample,
            512 if args.format == "sql" else train_config["max_seq_length"],
        )
        for sample in samples
    ]

    class SupervisedDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(encoded_samples)

        def __getitem__(self, index):
            return encoded_samples[index]

    model_kwargs = {"trust_remote_code": True}
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        train_config["model_id"], **model_kwargs
    )

    peft_model = get_peft_model(
        model,
        LoraConfig(
            r=lora_config["r"],
            lora_alpha=lora_config["lora_alpha"],
            target_modules=lora_config["target_modules"],
            lora_dropout=lora_config["lora_dropout"],
            bias=lora_config["bias"],
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
        ),
    )
    training_args_kwargs = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": train_config["per_device_train_batch_size"],
        "gradient_accumulation_steps": train_config["gradient_accumulation_steps"],
        "learning_rate": train_config["learning_rate"],
        "lr_scheduler_type": train_config["lr_scheduler_type"],
        "max_steps": train_config["max_steps"],
        "fp16": bool(train_config["fp16"] and torch.cuda.is_available()),
        "logging_steps": train_config["logging_steps"],
        "save_strategy": train_config["save_strategy"],
        "seed": train_config["seed"],
        "report_to": "none",
        "remove_unused_columns": False,
    }
    training_args_kwargs.update(warmup_arguments(TrainingArguments, train_config))
    training_args = TrainingArguments(
        **training_args_kwargs
    )
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=SupervisedDataset(),
        data_collator=lambda features: collate_training_batch(
            features, tokenizer.pad_token_id, tokenizer.padding_side
        ),
    )
    trainer.train()
    peft_model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved {args.format} LoRA adapter to {output_dir}")


if __name__ == "__main__":
    main()
