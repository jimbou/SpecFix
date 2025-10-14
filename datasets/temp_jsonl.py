import json

input_file = "/home/jim/SpecFix/datasets/humaneval+.jsonl"
output_file = "/home/jim/SpecFix/datasets/humaneval_final.jsonl"
limit = 100

with open(input_file, "r") as fin, open(output_file, "w") as fout:
    count = 0
    for line in fin:
        try:
            json.loads(line)
        except json.JSONDecodeError:
            continue  # skip invalid lines
        fout.write(line)
        count += 1
        if count >= limit:
            break

print(f"✅ Saved {count} valid entries to {output_file}")
