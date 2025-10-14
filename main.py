import argparse
from pathlib import Path

from requests import options
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
    #add output stat file argument which is not required
    parser.add_argument("-o", "--output_stat_file", type=str, default=None)
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional path for Mnimi persistent cache (disabled when omitted)",
    )
    parser.add_argument(
        "--cache-replication",
        action="store_true",
        help="Fail on cache miss instead of querying the model (replication mode)",
    )

    options = parser.parse_args()

    evaluator = SpecFixAccuracyEvaluator(
            differential_tester,
            ground_truth_tester,
            options.model,
            options.temperature,
            cache_dir=options.cache_dir,
            replication=options.cache_replication,
        )
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
                summarize_result(problem, repaired_requirement, original_clusters.serialize(), repaired_clusters.serialize(), original_result,
                                 repaired_result))
        else:
            results.append(summarize_result(problem, None, original_clusters.serialize(), None, original_result, None))

    with jsonlines.open(output_file, "w") as writer:
        writer.write_all(results)
    #if output stat file is provided, write the summary statistics to that file



    # ---- Infer stats from current results structure (no producer changes) ----

    def _safe_get(d, *keys, default=None):
        cur = d
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    def _cluster_metrics(cluster_obj):
        """Return (weighted_test_consistency, entropy) from a cluster dict or (None, None)."""
        if not isinstance(cluster_obj, dict):
            return None, None
        return (
            cluster_obj.get("weighted_test_consistency"),
            cluster_obj.get("entropy"),
        )

    def _is_detected(record):
        # In your pipeline, repaired_* are only populated when detection fired.
        return record.get("repaired_clusters") is not None

    def _is_improved(record):
        # If not detected, it can't be improved.
        if not _is_detected(record):
            return False

        res = record.get("result", {})
        opk = _safe_get(res, "original_passk")
        rpk = _safe_get(res, "repaired_passk")
        oavr = _safe_get(res, "original_avg_pass_rate")
        ravr = _safe_get(res, "repaired_avg_pass_rate")

        # 1) Prefer Pass@k if both present
        if opk is not None and rpk is not None:
            try:
                return float(rpk) > float(opk)
            except Exception:
                pass

        # 2) Fall back to average pass rate
        if oavr is not None and ravr is not None:
            try:
                return float(ravr) > float(oavr)
            except Exception:
                pass

        # 3) Fall back to cluster metrics: first improve consistency, then reduce entropy
        oWTC, oH = _cluster_metrics(record.get("original_clusters"))
        rWTC, rH = _cluster_metrics(record.get("repaired_clusters"))

        # Higher weighted test consistency is better
        if (rWTC is not None) and (oWTC is not None) and (rWTC > oWTC):
            return True
        # If consistency unchanged, lower entropy is better
        if (rWTC is not None) and (oWTC is not None) and (rWTC == oWTC):
            if (rH is not None) and (oH is not None) and (rH < oH):
                return True

        return False

    # Aggregate
    total = len(results)
    detected = sum(1 for r in results if _is_detected(r))
    improved = sum(1 for r in results if _is_improved(r))

    # (Optional) compute simple ratios safely
    detected_ratio = (detected / total) if total else 0.0
    improved_ratio = (improved / total) if total else 0.0

    # ---- Write summary (keep your existing usage stats if you have them) ----
    


    usage = evaluator.model.get_usage_stats()
    if options.output_stat_file:
        with open(options.output_stat_file, "w") as f:
            f.write(f"Total problems: {total}\n")
            f.write(f"SpecFix detected: {detected} ({detected_ratio:.2%})\n")
            f.write(f"SpecFix improved: {improved} ({improved_ratio:.2%})\n")

            #usage stats too
            f.write(f"Prompt tokens: {usage['prompt_tokens']}\n")
            f.write(f"Completion tokens: {usage['completion_tokens']}\n")
            f.write(f"Total tokens: {usage['total_tokens']}\n")
            f.write(f"API time (s): {usage['api_time_seconds']:.3f}\n")
        print(f"✅ Summary statistics saved to {options.output_stat_file}")
    # usage = evaluator.model.get_usage_stats()
    print("\nModel usage summary:")
    print(f"  Prompt tokens:     {usage['prompt_tokens']}")
    print(f"  Completion tokens: {usage['completion_tokens']}")
    print(f"  Total tokens:      {usage['total_tokens']}")
    print(f"  API time (s):      {usage['api_time_seconds']:.3f}")

if __name__ == "__main__":
    main()
