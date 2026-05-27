import argparse
import json
import os
import sys
import warnings
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore", category=SyntaxWarning)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_original(dataset_name, path):
    if dataset_name == "math":
        rows = load_jsonl(path)
        return [{"question": row["problem"], "solution": row["solution"]} for row in rows]

    if path.endswith(".jsonl"):
        rows = load_jsonl(path)
    else:
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
    return rows


def correctness_for_records(dataset_name, records, original_rows):
    from bon_eval_utils import eval_gsm8k, eval_math_prm

    if dataset_name == "math":
        problems = [original_rows[int(row["idx"])] for row in records]
        _, correct, outputs = eval_math_prm(records, all_problems=problems, is_extract=False)
        return np.array(correct, dtype=bool), outputs

    answers = [original_rows[int(row["idx"])]["answer"] for row in records]
    _, correct, outputs = eval_gsm8k(records, answers=answers, is_extract=True)
    return np.array(correct, dtype=bool), outputs


def rank_auc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    pos = labels.sum()
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)

    # Average tied ranks.
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        if end - start > 1:
            avg_rank = (start + 1 + end) / 2.0
            ranks[order[start:end]] = avg_rank
        start = end

    pos_rank_sum = ranks[labels].sum()
    return float((pos_rank_sum - pos * (pos + 1) / 2.0) / (pos * neg))


def corr(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if scores.std() == 0 or labels.std() == 0:
        return float("nan")
    return float(np.corrcoef(scores, labels)[0, 1])


def select_acc(groups, correct, key, reverse=True, n_values=(1, 8, 16, 32, 64, 128)):
    out = {}
    for n in n_values:
        hits = []
        for rows in groups.values():
            rows_n = rows[:n]
            picked = sorted(rows_n, key=key, reverse=reverse)[0]
            hits.append(correct[picked["_pos"]])
        out[n] = 100.0 * sum(hits) / len(hits)
    return out


def oracle_acc(groups, correct, n_values=(1, 8, 16, 32, 64, 128)):
    out = {}
    for n in n_values:
        hits = []
        for rows in groups.values():
            rows_n = rows[:n]
            hits.append(any(correct[row["_pos"]] for row in rows_n))
        out[n] = 100.0 * sum(hits) / len(hits)
    return out


def print_curve(name, values):
    print(name)
    for n, value in values.items():
        print(f"  @{n:<3} {value:6.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored-file", required=True, help="JSON file saved by bon_eval_hf.py")
    parser.add_argument("--dataset-name", choices=["math", "gsm8k"], required=True)
    parser.add_argument("--original-file", required=True, help="MATH-500 jsonl or GSM-Plus testmini jsonl/json")
    args = parser.parse_args()

    with open(args.scored_file, encoding="utf-8") as f:
        records = json.load(f)

    for pos, row in enumerate(records):
        row["_pos"] = pos
        row["reward"] = float(row["reward"])

    groups = defaultdict(list)
    for row in records:
        groups[int(row["idx"])].append(row)
    groups = dict(sorted(groups.items()))

    original_rows = load_original(args.dataset_name, args.original_file)
    correct, _ = correctness_for_records(args.dataset_name, records, original_rows)
    rewards = np.array([row["reward"] for row in records], dtype=float)

    per_question_ranges = []
    per_question_stds = []
    for rows in groups.values():
        values = np.array([row["reward"] for row in rows], dtype=float)
        per_question_ranges.append(values.max() - values.min())
        per_question_stds.append(values.std())

    print(f"records: {len(records)}")
    print(f"questions: {len(groups)}")
    print(f"responses/question: min={min(len(v) for v in groups.values())}, max={max(len(v) for v in groups.values())}")
    print(f"raw correctness: {100.0 * correct.mean():.2f}")
    print()
    print("reward distribution")
    print(f"  mean={rewards.mean():.6f} std={rewards.std():.6f} min={rewards.min():.6f} max={rewards.max():.6f}")
    print(f"  p01={np.percentile(rewards, 1):.6f} p50={np.percentile(rewards, 50):.6f} p99={np.percentile(rewards, 99):.6f}")
    print(f"  per-question range mean={np.mean(per_question_ranges):.6f} median={np.median(per_question_ranges):.6f}")
    print(f"  per-question std   mean={np.mean(per_question_stds):.6f} median={np.median(per_question_stds):.6f}")
    print()
    print("reward vs correctness")
    print(f"  correct reward mean={rewards[correct].mean():.6f}")
    print(f"  wrong reward mean  ={rewards[~correct].mean():.6f}")
    print(f"  corr={corr(rewards, correct):.6f}")
    print(f"  auc ={rank_auc(rewards, correct):.6f}")
    print()

    print_curve("select by highest reward", select_acc(groups, correct, key=lambda row: row["reward"], reverse=True))
    print_curve("select by lowest reward", select_acc(groups, correct, key=lambda row: row["reward"], reverse=False))
    print_curve("oracle pass@N", oracle_acc(groups, correct))


if __name__ == "__main__":
    main()
