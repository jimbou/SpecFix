import os
import subprocess
import sys
import types
import random
from datetime import datetime
from copy import deepcopy
import math
import re
import jsonlines
from func_timeout import func_timeout, FunctionTimedOut
from tqdm import trange
from solution_transformer import remove_comments_and_asserts, transform_code
from evalplus.data import get_human_eval_plus, get_mbpp_plus, get_human_eval_plus_hash, get_mbpp_plus_hash
from evalplus.evaluate import get_groundtruth
import numpy as np
from statistics import mean


def post_process(text: str) -> str:
    python_pattern = re.compile(r'```python\s*(.*?)\s*```', re.DOTALL)
    match = python_pattern.search(text)
    if match:
        return match.group(1)

    general_pattern = re.compile(r'```(.*?)```', re.DOTALL)
    match = general_pattern.search(text)
    if match:
        return match.group(1)
    return text.strip()


def execute(func_str, func_args, entry_point):
    max_install_attempts = 3
    installed_modules = set()
    if func_str == "":
        return "EmptyCodeError"
    while True:
        try:
            local_env = {}
            exec(func_str, local_env)

            if entry_point in local_env:
                func = local_env[entry_point]
            else:
                target_funcs = [f for f in local_env.values() if isinstance(f, types.FunctionType)]
                if len(target_funcs) == 1:
                    func = target_funcs[0]
                else:
                    func = random.choice(target_funcs)

            return func(*func_args)

        except (ModuleNotFoundError, ImportError) as e:
            module_name = e.name
            if module_name in installed_modules:
                return "ModuleNotFoundError"
            if len(installed_modules) >= max_install_attempts:
                return "ModuleNotFoundError"

            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", module_name], stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL)
                installed_modules.add(module_name)
                continue
            except subprocess.CalledProcessError:
                return "ModuleNotFoundError"

        except Exception as e:
            return e.__class__.__name__


def execute_inputs(func_str, inputs_list, entry_point, timeout=1):
    results = []
    for i in trange(len(inputs_list)):
        try:
            # results.append([execute(func_str, deepcopy_arguments(inputs_list[i]), entry_point)])
            deepcopy_argument = deepcopy(inputs_list[i])
            results.append(
                [func_timeout(timeout, execute, args=(func_str, deepcopy_argument, entry_point))])
        except FunctionTimedOut:
            results.append(["Timeout"])
    return results


def unwrap(string: str, label: str) -> str:
    pattern = re.compile(rf'<{label}>(.*?)</{label}>', re.DOTALL)
    match = pattern.search(string)

    extracted = match.group(1).strip() if match else string

    if label in {'code', 'test'} and '```' in extracted:
        extracted = post_process(extracted)

    if label == 'code':
        try:
            cleaned = remove_comments_and_asserts(extracted)
            return transform_code(cleaned).strip()
            # return cleaned.strip()
        except Exception as e:
            print("AST parsing error")
            print(extracted)
            return ""

    return extracted


def get_failed_input_output(result_list, inputs, outputs):
    if inputs == [] or outputs == [] or compare(result_list, outputs):
        return [], 1
    failed_input_output_examples = []
    for i in range(len(inputs)):
        if not compare(result_list[i], outputs[i]):
            failed_input_output_examples.append([inputs[i], result_list[i], outputs[i]])
    return failed_input_output_examples, 1 - (len(failed_input_output_examples) / len(inputs))


def compare(a, b):
    try:
        if a == "Timeout" or b == "Timeout":
            return True
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                return False
            for x, y in zip(a, b):
                if not compare(x, y):
                    return False
            return True
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return math.isclose(a, b, rel_tol=0.001)
        else:
            return a == b
    except:
        return False


def construct_output_file(cwd, model_name, dataset, task):
    # timestamp by minute
    time_stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    if not os.path.exists(f"{cwd}/{task}/{model_name}"):
        os.makedirs(f"{cwd}/{task}/{model_name}")
    output_file = f"{cwd}/{task}/{model_name}/{dataset}-{time_stamp}.jsonl"
    return output_file


def get_parameter_number(requirement, entry_point):
    for line in requirement.split("\n"):
        if f"def {entry_point}(" in line:
            return line.split("(")[1].split(")")[0].count(":")


def read_jsonl(file_name):
    with jsonlines.open(file_name) as reader:
        return list(reader)


def get_evalplus_inputs_outputs(data_name):
    data = get_human_eval_plus() if data_name == "humaneval" else get_mbpp_plus()
    hash = get_human_eval_plus_hash() if data_name == "humaneval" else get_mbpp_plus_hash()
    expected_outputs = get_groundtruth(data, hash, [])
    inputs = []
    outputs = []
    for key in data.keys():
        problem = data[key]
        inputs.append((problem['base_input'] + problem['plus_input']) if problem['plus_input'] != {} else problem[
            'base_input'])
        outputs.append([[output] for output in expected_outputs[key]['base'] + expected_outputs[key]['plus']])
    return inputs, outputs

def calculate_pass_k(n, c, k):
    if c == 0:
        return 0.0

    if (n - c) < k:
        return 1.0

    prob_no_pass = 1.0
    for i in range(k):
        prob_no_pass *= (n - c - i) / (n - i)

    return 1 - prob_no_pass


def calculate_test_consistency(program_str, entry_point, inputs, outputs):
    result_list = execute_inputs(program_str, inputs, entry_point)
    failed_input_output_examples, test_consistency = get_failed_input_output(result_list,
                                                                             inputs, outputs)
    return failed_input_output_examples, test_consistency


def get_exception_list():
    # list of major exception types
    exception_type = [["TypeError"], ["ValueError"], ["SyntaxError"], ["NameError"], ["IndexError"],
                      ["KeyError"], ["AttributeError"], ["ImportError"], ["ModuleNotFoundError"], ["MemoryError"],
                      ["RecursionError"], ["ZeroDivisionError"],
                      ["NotImplementedError"], ["RuntimeError"], ["AssertionError"], ["OverflowError"],
                      ["FloatingPointError"], ["IndentationError"]]
    return exception_type


def safe_eval(val):
    class ReMatch:
        def __init__(self, span, match):
            self.span = span
            self.match = match

        def __repr__(self):
            return f"<re.Match object; span={self.span}, match=<'{self.match}'>"

    def replace_func(m):
        start = int(m.group(1))
        end = int(m.group(2))
        text = m.group(3)
        return f"ReMatch(({start}, {end}), '{text}')"

    if "re.Match object" in val:
        pattern = r"<re\.Match object; span=\((\d+),\s*(\d+)\), match='([^']+)'>"
        val = re.sub(pattern, replace_func,
                     val)
    return eval(val, {
        "np": np, "inf": float("inf"),
        "nan": float("nan"),
        "ReMatch": ReMatch,
        "ZeroDivisionError": ZeroDivisionError,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "IndexError": IndexError,
        "KeyError": KeyError,
        "AttributeError": AttributeError,
        "NameError": NameError,
        "SyntaxError": SyntaxError,
        "AssertionError": AssertionError,
        "RecursionError": RecursionError,
        "FileNotFoundError": FileNotFoundError,
        "ModuleNotFoundError": ModuleNotFoundError,
        "ImportError": ImportError,
        "MemoryError": MemoryError,
        "OverflowError": OverflowError,
        "RuntimeError": RuntimeError,
        "StopIteration": StopIteration
    })


def summarize_result(problem, repaired_requirement, original_clusters, repaired_clusters, original_result,
                     repaired_result):
    if repaired_requirement is None:
        return {
            "task_id": problem["task_id"],
            "original_requirement": problem["requirement"],
            "repaired_requirement": None,
            "original_clusters": original_clusters,
            "repaired_clusters": None,
            "result": {
                "original_passk": original_result["passk"],
                "original_avg_pass_rate": original_result["avg_pass_rate"],
                "repaired_pass_rate": None,
                "original_nzpassk": original_result["passk"] > 0,
                "original_majority_passk": original_result["majority_passk"],
                "original_entropy": original_clusters["entropy"],
                "repaired_passk": None,
                "repaired_avg_pass_rate": None,
                "repaired_nzpassk": None,
                "repaired_majority_passk": None,
                "repaired_entropy": None
            }
        }
    return {
        "task_id": problem["task_id"],
        "original_requirement": problem["requirement"],
        "repaired_requirement": repaired_requirement,
        "original_clusters": original_clusters,
        "repaired_clusters": repaired_clusters,
        "result": {
            "original_passk": original_result["passk"],
            "original_avg_pass_rate": original_result["avg_pass_rate"],
            "original_nzpassk": original_result["passk"] > 0,
            "original_majority_passk": original_result["majority_passk"],
            "original_entropy": original_clusters["entropy"],
            "repaired_passk": repaired_result["passk"],
            "repaired_avg_pass_rate": repaired_result["avg_pass_rate"],
            "repaired_nzpassk": repaired_result["passk"] > 0,
            "repaired_majority_passk": repaired_result["majority_passk"],
            "repaired_entropy": repaired_clusters["entropy"]
        }
    }
