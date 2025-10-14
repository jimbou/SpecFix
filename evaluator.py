import ast
import concurrent.futures
import math
from time import sleep

from prompting import *
from model import Model
from typing import Optional
from utils import unwrap, get_parameter_number, execute_inputs, compare, get_failed_input_output, \
    calculate_pass_k


class SpecFixAccuracyEvaluator:
    def __init__(self, differential_tester=None, ground_truth_tester=None, model="gpt-4o",
                 temperature=None, cache_dir: Optional[str] = None, replication: bool = False):
        self.differential_tester = differential_tester
        self.ground_truth_tester = ground_truth_tester
        self.model = Model(model, temperature, cache_dir=cache_dir, replication=replication)
        self.temperature = temperature

    def get_clusters(self, requirement, programs, test_inputs, entry_point, examples=None):
        print("GET CLUSTERS")
        clusters = self.differential_tester(programs, test_inputs, entry_point)
        clusters.set_requirement(requirement)
        clusters.set_entry_point(entry_point)
        clusters.set_input_output_examples(examples)
        return clusters

    def get_test_consistency(self, clusters):
        print("CALCULATE TEST CONSISTENCY")
        self.ground_truth_tester(clusters)

    def parallel_generate_programs(self, requirement, entry_point, n_programs, max_workers=10):
        generated_programs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.generate_program, requirement, entry_point)
                       for _ in range(n_programs)]
            for future in concurrent.futures.as_completed(futures):
                prog = future.result()
                generated_programs.append(prog)
        generated_programs = [prog for prog in generated_programs if prog != ""]
        return generated_programs

    def generate_programs(self, requirement, entry_point, n_programs, cache_mode: str = "independent"):
        if "deepseek" in self.model.model_name:
            batch_size = 5
            generated_programs = []
            for _ in range(math.ceil(n_programs / batch_size)):
                response = self.model.get_response_sample(instruction_generate_code,
                                                          prompt_generate_code(requirement, entry_point), batch_size,
                                                          cache_mode=cache_mode)
                generated_programs.extend([unwrap(prog, "code") for prog in response])
            if len(generated_programs) > n_programs:
                generated_programs = generated_programs[: n_programs]
            generated_programs = [prog for prog in generated_programs if prog != ""]
            return generated_programs
        if "gpt" in self.model.model_name:
            response = self.model.get_response_sample(instruction_generate_code,
                                                      prompt_generate_code(requirement, entry_point), n_programs,
                                                      cache_mode=cache_mode)
            generated_programs = [unwrap(prog, "code") for prog in response]
            generated_programs = [prog for prog in generated_programs if prog != ""]
            return generated_programs
        else:
            return self.parallel_generate_programs(requirement, entry_point, n_programs)

    def generate_program(self, requirement, entry_point):
        for i in range(5):
            try:
                print("GENERATE PROGRAM ATTEMPT", i)
                response = self.model.get_response(
                    instruction_generate_code,
                    prompt_generate_code(requirement, entry_point),
                    cache_mode="independent",
                )
                code = unwrap(response, "code")
                if code == "":
                    raise Exception
                return code
            except Exception as e:
                print(e)
                sleep(1)
                continue
        print("GENERATE PROGRAM FAILED")
        return ""

    def generate_tests(self, requirements, entry_point):
        for i in range(10):
            print("GENERATE TEST ATTEMPT", i)
            tests = []
            para_number = get_parameter_number(requirements, entry_point)
            try:
                response = self.model.get_response(
                    instruction_generate_test,
                    prompt_generate_test(requirements, entry_point, para_number),
                    cache_mode="independent",
                )
                response = unwrap(response, "tests")
                for line in response.splitlines():
                    test = ast.literal_eval("[" + unwrap(line, "test") + "]")
                    if len(test) == para_number:
                        tests.append(test)
                    if len(tests) > 50:
                        break
                if len(tests) == 0:
                    raise Exception
                return tests
            except Exception as e:
                print(e)
                continue
        print("GENERATE TEST FAILED")
        return []

    def vanilla_repair_requirements(self, requirements):
        print("VANILLA REPAIR REQUIREMENTS")
        response = self.model.get_response(
            instruction_vanilla_repair,
            prompt_vanilla_repair(requirements),
            cache_mode="persistent",
        )
        return unwrap(response, "requirement")

    def classification(self, requirements):

        print("CLASSIFICATION")
        response = self.model.get_response(
            instruction_classification,
            prompt_classification(requirements),
            cache_mode="persistent",
        )
        answer = unwrap(response, "answer")
        reason = unwrap(response, "reasoning")
        if answer == "Yes" or answer == "No":
            return answer, reason
        return None

    def contrastive_inference(self, requirement, entry_point, specified_programs, other_programs, diff_outputs):

        print("CONTRASTIVE INFERENCE")
        response = self.model.get_response(
            instruction_contrastive_inference,
            prompt_contrastive_inference(
                requirement,
                entry_point,
                specified_programs,
                other_programs,
                diff_outputs,
            ),
            True,
            cache_mode="repeatable_attempt",
        )
        repaired_requirement = unwrap(response, "requirement")
        if repaired_requirement != "":
            return repaired_requirement
        return None

    def evaluate(self, requirement, clusters, inputs, outputs, entry_point, k, sample):
        if requirement is None:
            return None, None, [], []
        passes = 0
        programs = self.generate_programs(requirement, entry_point, sample)
        if len(programs) == 0:
            return None, None, [], []
        pass_rates = []
        for i in range(len(programs)):
            passed = False
            program = programs[i]
            result = execute_inputs(program, inputs, entry_point)
            if compare(result, outputs):
                passed = True
                pass_rates.append(1)
            else:
                failed_input_output, pass_rate = get_failed_input_output(result, inputs, outputs)
                # print(f"Failed input-output: {failed_input_output}, pass rate: {pass_rate}")
                pass_rates.append(pass_rate)
            passes += int(passed)
        return {
            "passk": calculate_pass_k(len(programs), passes, k),
            "avg_pass_rate": sum(pass_rates) / len(pass_rates),
            "majority_passk": self.solved_with_majority_vote(clusters, inputs, outputs)
        }

    def specfix_detect(self, problem, n_programs, label=None):
        if label is None:
            requirement, entry_point, examples, task_id = problem['requirement'], problem['entry_point'], problem[
                'input_output_examples'], problem['task_id']
        else:
            requirement, entry_point, examples, task_id = problem[label], problem['entry_point'], problem[
                'input_output_examples'], problem['task_id']
        print(F"SPECFIX DETECT {task_id}")
        test_inputs = self.generate_tests(requirement, entry_point)
        programs = self.generate_programs(requirement, entry_point, n_programs)
        if len(programs) == 0:
            return False, None
        clusters = self.get_clusters(requirement, programs, test_inputs, entry_point, examples)
        self.get_test_consistency(clusters)
        if not (clusters.entropy == 0 and clusters.weighted_test_consistency in {1, -1}):
            return True, clusters
        return False, clusters

    def specfix_repair(self, clusters, n_programs):
        requirement = clusters.requirement
        entry_point = clusters.entry_point
        examples = clusters.input_output_examples
        test_inputs = clusters.llm_generated_inputs

        for repair_attempts in range(3):
            repair_method, largest_cluster = clusters.select_repair_method()

            if repair_method == 0:
                repaired_program = self.program_repair(
                    requirement, entry_point,
                    largest_cluster.programs_str[0],
                    largest_cluster.failed_input_output_examples
                )
                repaired_requirement = self.contrastive_inference(
                    requirement, entry_point, repaired_program,
                    [largest_cluster.programs_str[0]],
                    largest_cluster.failed_input_output_examples
                )
            else:
                other_clusters, diff_outputs = clusters.get_other_clusters_and_diff_outputs(largest_cluster)
                other_programs = [cluster.get_min_length_program() for cluster in other_clusters]
                repaired_requirement = self.contrastive_inference(
                    requirement, entry_point,
                    largest_cluster.programs_str[0], other_programs, diff_outputs
                )

            repaired_programs = self.generate_programs(
                repaired_requirement,
                entry_point,
                n_programs,
                cache_mode="independent",
            )
            repaired_clusters = self.get_clusters(
                repaired_requirement, repaired_programs,
                test_inputs, entry_point, str(examples)
            )
            self.get_test_consistency(repaired_clusters)

            if repaired_clusters.entropy == 0 and repaired_clusters.weighted_test_consistency == 1:
                return repaired_requirement, repaired_clusters

            if (repaired_clusters.weighted_test_consistency > clusters.weighted_test_consistency or
                    (repaired_clusters.weighted_test_consistency == clusters.weighted_test_consistency and
                     repaired_clusters.entropy < clusters.entropy)):
                requirement, clusters = repaired_requirement, repaired_clusters

        return requirement, clusters

    def program_repair(self, requirement, entry_point, program, failed_input_output_examples):

        print("PROGRAM REPAIR")
        response = self.model.get_response(
            instruction_program_repair,
            prompt_program_repair(
                requirement,
                entry_point,
                program,
                failed_input_output_examples,
            ),
            cache_mode="repeatable_attempt",
        )
        repaired_program = unwrap(response, "code")
        return repaired_program

    def solved_with_majority_vote(self, clusters, inputs, outputs):
        if clusters is None:
            return None
        cluster = max(clusters.cluster_list, key=lambda c: c.probability)
        program = cluster.programs_str[0]
        result = execute_inputs(program, inputs, clusters.entry_point)
        if compare(result, outputs):
            return True
        return False
