# CMAD

## Overview

We provide the code of our paper. The algorithm implementation code is in the `CMAD` folder, and the experimental code is in the `experiments` folder.

## Quick Start

### Install packages

```bash
conda create -n cmad python=3.10
conda activate cmad
pip install -r requirements.txt
```

### Add API keys in `template.env` and change its name to `.env`

```python
BASE_URL = "" # the BASE_URL of OpenAI LLM backend
API_KEY = "" # for OpenAI LLM backend
```

### Download Datasets

Download MMLU, HumanEval and GSM8K datasets from MMLU, HumanEval and GSM8K. And put them in different folders.

### Code Expalin

cold_start:  runs the baseline multi-agent solver and logs full transcripts/graphs for MMLU.
generate_false/ture_conterfactual.py: creates counterfactual runs by muting selected edges to produce training signal from outcome changes.
trainscorer.py: TrainScorer trains a frozen teacher that scores each message/edge for how much it helps or hurts correctness.
run.py: Run trains the Adaptive-K edge selector with the teacher and then evaluates dynamic edge selection on MMLU.







