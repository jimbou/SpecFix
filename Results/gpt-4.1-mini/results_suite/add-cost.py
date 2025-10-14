# add_cost.py
# Usage: python add_cost.py input.txt

import sys
import re

def main(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    prompt_tokens = 0
    completion_tokens = 0

    # Extract token values from the file
    for line in lines:
        if line.startswith("Prompt tokens:"):
            prompt_tokens = int(re.findall(r"\d+", line)[0])
        elif line.startswith("Completion tokens:"):
            completion_tokens = int(re.findall(r"\d+", line)[0])

    # Calculate total cost
    total_cost = (0.8 * prompt_tokens + 3.2 * completion_tokens) / 1_000_000

    # Append total cost line
    lines.append(f"Total cost: ${total_cost:.6f}\n")

    # Write back to file
    with open(file_path, 'w') as f:
        f.writelines(lines)

    print(f"Added total cost: ${total_cost:.6f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python add_cost.py <file.txt>")
    else:
        main(sys.argv[1])
