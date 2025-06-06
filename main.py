import argparse
from pathlib import Path
import jsonlines

from evaluator import SpecFixAccuracyEvaluator
from utils import (
    get_evalplus_inputs_outputs,
    construct_output_file,
    read_jsonl,
    summarize_result,
)
from tester import differential_tester, ground_truth_tester


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True, help="Dataset: humaneval and mbpp")
    parser.add_argument("-p", "--path", required=True, help="Dataset Path")
    parser.add_argument("-c", "--cluster_sample_size", type=int, default=20)
    parser.add_argument("-e", "--evaluation_sample_size", type=int, default=10)
    parser.add_argument("-k", "--passk", type=int, default=1, help="Pass@k value for evaluation")
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-t", "--temperature", type=float, default=None)
    options = parser.parse_args()

    evaluator = SpecFixAccuracyEvaluator(differential_tester, ground_truth_tester, options.model, options.temperature)

    inputs, outputs = get_evalplus_inputs_outputs(options.dataset)

    output_file = construct_output_file(Path(__file__).resolve().parent, options.model, options.dataset,
                                        "Results")
    results = []
    problems = read_jsonl(options.path)
    for index, problem in enumerate(problems):
        detect_result, original_clusters = evaluator.specfix_detect(problem, options.cluster_sample_size)
        original_result = evaluator.evaluate(
            problem["requirement"], original_clusters, inputs[index], outputs[index], problem["entry_point"],
            options.passk,
            options.evaluation_sample_size
        )
        if detect_result:
            repaired_requirement, repaired_clusters = evaluator.specfix_repair(original_clusters,
                                                                               options.cluster_sample_size)
            repaired_result = evaluator.evaluate(
                repaired_requirement, repaired_clusters, inputs[index], outputs[index], problem["entry_point"],
                options.passk,
                options.evaluation_sample_size
            )
            results.append(
                summarize_result(problem, repaired_requirement, original_clusters, repaired_clusters, original_result,
                                 repaired_result))
        else:
            results.append(summarize_result(problem, None, original_clusters, None, original_result, None))

    with jsonlines.open(output_file, "w") as writer:
        writer.write_all(results)


if __name__ == "__main__":
    main()
