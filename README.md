# Measuring the Contributions of LoRA, Template-Based Rationale, and Execution-Guided Repair for Text-to-SQL with a 1.5B Model

![Python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)
![PyTorch](https://img.shields.io/badge/Made%20with-PyTorch-ee4c2c.svg)
![Open Source](https://img.shields.io/badge/Open%20Source-Yes-brightgreen.svg)
![Model](https://img.shields.io/badge/Backbone-Qwen2.5--Coder--1.5B-blue.svg)

## Abstract

Translating natural language questions into executable SQL queries (Text-to-SQL) remains challenging for small language models (1B–2B parameters), particularly under constrained labeled data budgets. While parameter-efficient fine-tuning (LoRA), rationale generation, and execution-guided repair are widely adopted techniques, their isolated and combined effects on small models with limited training data require controlled measurement. 

In this work, we present a controlled empirical study evaluating a lightweight **LoRA-CoT-Repair** pipeline built on **Qwen2.5-Coder-1.5B-Instruct** under a strict training budget of 800 examples from the Spider benchmark (with full-dataset training included as a reference). Our pipeline combines LoRA adaptation (~18.5M trainable parameters), a structured template-based rationale acting as a schema-linking scratchpad, and a single-pass execution-guided repair step triggered only upon SQLite runtime errors. Experiments on the full Spider development set (1,034 examples) show that LoRA accounts for the dominant share of performance gains (+289 solved queries over zero-shot), while template-based rationale (+23 queries) and single-pass repair (+16 queries) provide smaller, complementary improvements, reaching **61.3% Execution Accuracy (EX)** at 800 training examples and **68.9% EX** when scaled to the full training set.

*(Note: This work focuses on controlled component attribution under resource constraints and does not claim superior data efficiency over existing methods or true multi-step reasoning capabilities.)*

## Citation

```bibtex
@article{trinh2026loracotrepair,
  title={Data-Efficient Text-to-SQL with Small Language Models: Combining LoRA, Chain-of-Thought Generation, and Execution-Guided Repair},
  author={Tran, Thi Dieu-Trinh and Le, Quang-Hung},
  year={2026}
}
```
