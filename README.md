# Vertical LLM in Legal Domain

## Introduction
Verticalization refers to specializing a general-purpose LM for a specific domain or task. Vertical LLMs aim to overcome the limitations of general models in specific contexts. In this experiment, `legal-bert-base-uncased` model , which is vertically pre-trained on English legal data, is fine-tuned and evaluated on `scotus` dataset of LexGLUE benchmark.

## Getting Started
Python 3 is required to run the scripts.

### Installation
Following packages are required to be installed:
- transformers
- datasets
- evaluate
- torch
- numpy

```
pip install transformers datasets torch numpy evaluate
```

## LEGAL-BERT
[Legal-BERT](https://huggingface.co/nlpaueb/legal-bert-base-uncased) is a family of BERT models for the legal domain.

## LexGLUE
[LexGLUE](https://huggingface.co/datasets/coastalcph/lex_glue) is a benchmark dataset for legal NLP tasks
