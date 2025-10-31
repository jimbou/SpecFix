# Automated Repair of Ambiguous Problem Descriptions for LLM-Based Code Generation

Prompting is the new programming. Prompts suffer from ambiguity of natural language, which often results in the generation of incorrect programs, therefore we need approaches that would help users detect and eliminate this ambiguity. SpecFix is the first approach that automatically repairs ambiguity in programming problem descriptions. Specifically, it minimally modifies the requirements to reduce code generation uncertainty and better aligning natural language with input-output examples. To do it precisely, SpecFix decomposes this task into two simpler steps: (1) analyzing and repairing the LLM's interpretation of the description - captured by the distribution of programs it induces - using traditional testing and program repair, and (2) refining the description based on distribution changes via a method we call contrastive specification inference. An extensive evaluation with four state-of-the-art LLMs (GPT-4o, GPT-4o-mini, DeepSeek-V3, and Qwen2.5-Coder-32B-Instruct) on three popular code generation benchmarks (HumanEval+, MBPP+ and LiveCodeBench) shows that SpecFix significantly increases Pass@1 of the modified requirements, and its repairs generalize across models.

**Automated Repair of Ambiguous Problem Descriptions for LLM-Based Code Generation**<br>
Haoxiang Jia, Robbie Morris, He Ye, Federica Sarro, Sergey Mechtaev<br>
International Conference on Automated Software Engineering, ASE 2025<br>
https://arxiv.org/abs/2505.07270

## Structure
The repository is structured as follows:
```
specfix/
    ├── main.py                 # Main script to run the tool
    ├── cluster.py              # Clustering functions for program clustering
    ├── evaluator.py            # Evaluation functions for repairing and measuring 
    ├── model.py                # Model functions for interacting with LLMs
    ├── prompt.py               # Prompts for each task
    ├── solution_transformer.py # Functions for transforming generated programs
    ├── testers.py              # Test functions for detecting ambiguity
    ├── utils.py                # Utility functions for various tasks
    ├── datasets/               # Dataset (HumanEval+ and MBPP+)
    ├── Results/                # Directory to save results
    ├── experiment_results/     # Directory for our experiment results
    ├── requirements.txt        # Python package dependencies
    └── README.md               # Documentation for the tool
```

## Installation
1. To install SpecFix, create a virtual environment and install the required packages:

```bash
python -m venv specfix-venv
source specfix-venv/bin/activate  # On Windows use `specfix-venv\Scripts\activate`
pip install -r requirements.txt
```

2. Set up the LLM API keys in the environment variables:
```bash
export LLM_API_KEY="your_llm_api_key"
```

## Usage
1. Run the tool:
```bash
cd specfix
python main.py \
  -d <dataset_name> \
  -p <path_to_dataset> \
  -c <clustering_sample_size> \
  -e <evaluation_sample_size> \
  -k <pass@k_value> \
  -m <model_name> \
  -t <temperature> \
  [--cache-dir <mnimi_cache_path>] \
  [--cache-replication]
```

* `--cache-dir` (optional) enables Mnimi's persistent cache by pointing to a directory where responses should be stored.
* `--cache-replication` (optional) toggles Mnimi's replication mode so runs fail fast on cache miss instead of calling the live model.

2. The results will be saved in the `Results` directory. The directory structure will be as follows:
```
Results/model_name/dataset_name/
    ├── humaneval-{timestamp}.jsonl
    └── mbpp-{timestamp}.jsonl
```
The jsonl files contain the following fields:
- `original_requirement`: The original requirement text.
- `repaired_requirement`: The repaired requirement text.
- `original_clusters`: The clusters of programs generated from the original requirement.
- `repaired_clusters`: The clusters of programs generated from the repaired requirement.
- `results`: 
  - `original_passk`: The pass@k value for the original requirement.
  - `original_avg_pass_rate`: The average pass rate for the original requirement.
  - `original_nzpassk`: The number of non-zero pass@k values for the original requirement.
  - `original_majority_passk`: The majority vote pass@k value for the original requirement.
  - `original_entropy`: The semantic entropy of the original requirement.
  - `repaired_passk`: The pass@k value for the repaired requirement.
  - `repaired_avg_pass_rate`: The average pass rate for the repaired requirement.
  - `repaired_nzpassk`: The number of non-zero pass@k values for the repaired requirement.
  - `repaired_majority_passk`: The majority vote pass@k value for the repaired requirement.
  - `repaired_entropy`: The semantic entropy of the repaired requirement.

## Example
To run the tool on the `HumanEval+` dataset with 20 samples for clustering and Pass@1 with 10 samples for evaluation, using the `gpt-4o` model with a temperature of 0.7, you can use the following command:

```bash
python main.py -d humaneval -p path/to/humaneval+.jsonl -c 20 -e 10 -k 1 -m gpt-4o -t 0.7 \
  --cache-dir ~/.mnimi/specfix
```

To rerun using only cached completions, add `--cache-replication` to the command above.


## Mnimi cache integration

SpecFix pairs naturally with [Mnimi](cached_llm.py), a lightweight LLM caching layer that keeps retries inexpensive while
preserving deterministic debugging and entropy calculation trhough independence when needed. For a step-by-step walkthrough showing how to wire Mnimi into the SpecFix pipeline, see
[`mnimi_integration.md`](mnimi_integration.md).

