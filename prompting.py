instruction_generate_code = "You are an assistant that generates Python code based on requirement."


def prompt_generate_code(requirement, entry_point):
    return f"""
Here is the given programming problem to solve.
{requirement}
Please implement the `{entry_point}` function and make sure that it matches the signature and functionality described in the requirement. 
Ensure to include necessary imports for function signature and function body.
Don't output any explanation or comments, only the function implementation.
Think step by step and wrap all generated code in <code></code> tags.
"""


instruction_generate_test = "You are an assistant that generates Python code inputs based on requirement."


def prompt_generate_test(requirement, entry_point, para_number):
    return f"""
    {requirement}
Given a requirement containing a function signature and docstring, your task is to generate inputs for function {entry_point} to cover all functionalities, including normal cases and corner cases.
Ensure the type and number of argument are matching the function signature. In this requirement, the argument number is {para_number}.
Don't output the function name, only the test inputs. If there are multiple arguments, separate them with commas.
Think step by step and wrap each test input in <test></test> tags and all test inputs in <tests></tests> tags. 
"""


instruction_classification = "You are an assistant that classifies the requirement whether it is ambiguous or not."


def prompt_classification(requirement):
    return f"""
Are the requirement ambiguous, i.e. leave room for multiple reasonable interpretations or contain contradictions, when considering the intended functionality? In your evaluation, consider how the program is expected to handle edge cases like extreme values. Exclude considerations related to handling invalid inputs or addressing aspects unrelated to functionality, such as performance.

1. If the requirement is ambiguous, answer "Yes".
2. If the requirement is unambiguous, answer "No".
4. Provide Your step-by-step reasoning for your judgment.

Format your final response in the following tags:
<answer>Yes or No</answer>
<reasoning>Your step-by-step reasoning</reasoning>

# Requirement
{requirement}
"""


instruction_vanilla_repair = "You are an assistant that repairs ambiguous requirements."


def prompt_vanilla_repair(requirement):
    return f"""
Given an ambiguous requirement, repair the requirement to remove ambiguity. 
{requirement}

Format your final repaired requirement with Python function syntax with type hints and a concise docstring, wrapped in <requirement></requirement> tags. 
<requirement>
def function_name(argument: type hint):->type hint 
        \"\"\"repaired requirement\"\"\"
</requirement>
"""


instruction_contrastive_inference = "You are an assistant that repairs ambiguous requirements based on the contrastive inference."


def prompt_contrastive_inference(requirement, entry_point, specified_programs, other_programs,
                                 diff_outputs):
    tests_str = ""
    for i, diff_output in enumerate(diff_outputs):
        tests_str += f"### Test {i + 1}\nInput: {diff_output[0]}\nExpected Output: {diff_output[2]}\n"
    return f"""
You are provided with:
1. An ambiguous description of a code generation task involving the function `{entry_point}`, which has led to multiple interpretations and consequently different generated implementations.
{requirement}
2. Selected implementation, reflecting the intended behavior.
{specified_programs}
3. Incorrect implementations generated from the ambiguous description, demonstrating alternative behaviors.
{other_programs}
4. Input-output examples explicitly stated in the requirement and the incorrect output produced by the program:
{tests_str}

You are tasked with repairing ambiguities in code-generation task requirements involving the function `{entry_point}` that have led to incorrectly generated code.
You will precisely repair the ambiguity in the requirement.

1. Carefully analyze the provided specification, identifying and clearly stating the specific wording or phrases that could be interpreted in multiple ways.
2. Analyze the selected program and selected outputs to determine the intended functionality and behavior.
3. Analyze the rejected implementation and rejected outputs to determine the unintended functionality and behavior.
4. State the potential sources of ambiguity that led to the divergence in outputs.
5. Concisely revise the requirement to remove ambiguity, aligning with behaviors of selected program and diverging from behaviors of rejected programs

Format the revised requirement explicitly in Python function syntax with type hints and a docstring, wrapped in <requirement></requirement> tags.
"""


instruction_program_repair = "You are an assistant that repairs ambiguous requirement by analyzing execution."


def prompt_program_repair(requirement, entry_point, program, failed_input_output_examples):
    tests = ""
    for i, (inp, output, canonical_output) in enumerate(failed_input_output_examples):
        inp = str(inp)[1:-1]
        output = str(output)[1:-1]
        canonical_output = str(canonical_output)[1:-1]
        tests += f"### Test {i + 1}\nInput: {inp}\nActual Output: {output}\nExpected Output: {canonical_output}\n"

    return f"""
You are tasked with repairing ambiguities in code-generation task requirements involving the function `{entry_point}` that have led to incorrectly generated code.
You will analyze the execution step by step to precisely repair the ambiguity in the requirement.

Given:
An ambiguous requirement:
{requirement}
Faulty program based on the ambiguous requirement:
{program}
Input-output examples explicitly stated in the requirement and the incorrect output produced by the program:
{tests}

1. Carefully analyze the task requirement to understand the intended behavior of the faulty program.
2. Examine the provided test cases, comparing the actual output with the expected output to clearly identify the underlying issue(s) such as logic errors, incorrect calculations, edge-case mishandling, or syntax issues.
3. Fix the Python function, ensuring the revised code passes all the provided test cases by generating the correct outputs

Format your repaired program with original function signature with type hints, wrapped in <code></code> tags.
"""
