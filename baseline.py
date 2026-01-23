import json

deon_path = "deontology_output.jsonl"
util_path = "utilitarian_output.jsonl"

def calculate_baseline(path):
    falses = 0
    trues = 0

    with open(path) as f:
        for line in f:
            record = json.loads(line)
            if record["messages"][1]["action"] == 0:
                falses += 1
            else:
                trues += 1
    print(f"{path}, falses: {falses}, trues: {trues}")

if __name__ == "__main__":
    calculate_baseline(deon_path)
    calculate_baseline(util_path)

#deontology_output.jsonl, falses: 51, trues: 24
#utilitarian_output.jsonl, falses: 27, trues: 48