# Few-Shot Language Model Evaluation Framework

![Version](https://img.shields.io/badge/version-v0.4.0-blue)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10256836.svg)](https://doi.org/10.5281/zenodo.10256836)

## Overview

This repository contains a framework for evaluating few-shot language models using Bayesian techniques to account for epistemic uncertainty. The framework samples from the last layer of a neural network LLM, producing multiple answers that reflect the model's uncertainty and confronts these samples to generate a comprehensive evaluation.

## Features

- **Few-Shot Evaluation**: Evaluates with minimal training examples.
- **Bayesian Sampling**: Reflects model's epistemic uncertainty.
- **Comprehensive Analysis**: Detailed evaluation of sampled outputs.

## Installation

For the installation, you can directly clone this repo.

## Usage

Select the model you want to use inside the class `custom_eval.py`:

```bash
import lm_eval
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval.models.huggingface import HFLM

# model_name = 'gpt2-large'
model_name = 'microsoft/phi-2'
# model_name = 'meta-llama/Llama-2-7b-chat-hf'

lm_obj = HFLM(pretrained=model_name, batch_size=32)

lm_eval.tasks.initialize_tasks()

results = lm_eval.simple_evaluate(
    model=lm_obj,
    tasks=["zero_shot_truthfulqa_mc1"],
    num_fewshot=0,
    device='cuda:0',
)

print(results["results"])
```

Then, you can run:

```bash
python custom_eval.py
```

## Sampling Technique

Modify in `lm_eval/evaluator.py` within the `simple_evaluate` method:

```bash
task_obj.model = lm
# task_obj.softmax_intro_k = 3
task_obj.softmax_intro_k = None
# task_obj.bayesian_intro_k = None
task_obj.bayesian_intro_k = 3
task_obj.H = None
task_obj.hessian_loader = get_hessian_dataloader(task_obj.model.tokenizer)
task_obj.max_length_hessian_loader = max_length_hessian_loader
task_obj.batch_size_hessian_loader = batch_size_hessian_loader
task_obj.M = M
task_obj.D_tokens = 21 * 10**9
task_obj.prior_precision = 10 ** 3
```

## Authors

- Nicolò Felicioni
- Riccardo Ruggieri

## Licence

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments

Special thanks to Zenodo for hosting the framework. For more details, visit the Zenodo page.
