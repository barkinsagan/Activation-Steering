# References

## Papers
<!-- - [Author, "Title", Year](URL) — one-line relevance note -->

## Tools & Libraries
- [transformers](https://github.com/huggingface/transformers) — model loading and tokenization
- [PyTorch](https://pytorch.org) — tensor ops and forward hooks
- [pandas](https://pandas.pydata.org) — result aggregation and CSV I/O

## Datasets
- **ML Textbook Completion Dataset** (custom, internal) — prompt-completion pairs extracted from an ML textbook. Used for both steering prompt construction and evaluation. Formatted as open completion and multiple-choice (MCQ). Columns: prompt, target [, false1, false2, false3].

## Models
- **meta-llama/Meta-Llama-3-8B** — primary model for all experiments
