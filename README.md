# Automated Repair of Ambiguous Natural Language Requirements

SpecFix is a tool for automatically repairing ambiguous natural language requirements to improve code generation by large language models (LLMs).

## Key Features
**Analyzing the distribution of programs** induced by a given requirement.

**Measuring and reducing semantic entropy**, which captures how many distinct interpretations (clusters of semantically equivalent programs) the requirement allows.

**Ensuring example consistency**, a novel metric that quantifies how well sampled programs satisfy the clarifying examples attached to the requirement.

**Performing contrastive specification inference**, which takes the repaired (or clustered) programs and iteratively refines the original text so that the most desirable interpretations are prioritized.

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
python main.py -d <dataset_name> -p <path_to_dataset> -c <clustering_sample_size> -e <evaluation_sample_size> -k <pass@k_value> -m <model_name> -t <temperature>
```

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
python main.py -d humaneval -p path/to/humaneval+.jsonl -c 20 -e 10 -k 1 -m gpt-4o -t 0.7
```
