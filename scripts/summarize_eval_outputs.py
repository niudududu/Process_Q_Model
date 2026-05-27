import argparse
import json
import os
import sys
import warnings
from collections import defaultdict

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)
warnings.filterwarnings("ignore", category=SyntaxWarning)

from analyze_scored_rewards import (
    corr,
    correctness_for_records,
    load_original,
    oracle_acc,
    rank_auc,
    select_acc,
)


POLICIES = ("metamath-mistral", "muggle", "llama3-70b-inst")
N_VALUES = (1, 8, 16, 32, 64, 128)


def load_records(path):
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    for pos, row in enumerate(records):
        row["_pos"] = pos
        row["reward"] = float(row["reward"])
    return records


def group_by_idx(records):
    groups = defaultdict(list)
    for row in records:
        groups[int(row["idx"])].append(row)
    return dict(sorted(groups.items()))


def summarize_one(dataset_name, policy, scored_file, original_rows):
    records = load_records(scored_file)
    groups = group_by_idx(records)
    correct, _ = correctness_for_records(dataset_name, records, original_rows)
    rewards = np.array([row["reward"] for row in records], dtype=float)

    bon = select_acc(groups, correct, key=lambda row: row["reward"], reverse=True, n_values=N_VALUES)
    low = select_acc(groups, correct, key=lambda row: row["reward"], reverse=False, n_values=N_VALUES)
    oracle = oracle_acc(groups, correct, n_values=N_VALUES)

    per_question_ranges = []
    for rows in groups.values():
        values = np.array([row["reward"] for row in rows], dtype=float)
        per_question_ranges.append(values.max() - values.min())

    return {
        "dataset": dataset_name,
        "policy": policy,
        "questions": len(groups),
        "responses_per_question": f"{min(len(v) for v in groups.values())}-{max(len(v) for v in groups.values())}",
        "raw": 100.0 * correct.mean(),
        "bon": bon,
        "low128": low[128],
        "oracle128": oracle[128],
        "auc": rank_auc(rewards, correct),
        "corr": corr(rewards, correct),
        "reward_std": float(rewards.std()),
        "q_range": float(np.mean(per_question_ranges)),
    }


def fmt(value):
    if value is None:
        return "-"
    if isinstance(value, float) and np.isnan(value):
        return "nan"
    return f"{value:.2f}"


def render_markdown(rows):
    headers = [
        "Dataset",
        "Policy",
        "Q",
        "Resp/Q",
        "Raw",
        "BON@1",
        "BON@8",
        "BON@16",
        "BON@32",
        "BON@64",
        "BON@128",
        "Low@128",
        "Oracle@128",
        "AUC",
        "Corr",
        "RewardStd",
        "QRange",
    ]
    lines = [
        "# Evaluation Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [
            row["dataset"],
            row["policy"],
            str(row["questions"]),
            row["responses_per_question"],
            fmt(row["raw"]),
            fmt(row["bon"][1]),
            fmt(row["bon"][8]),
            fmt(row["bon"][16]),
            fmt(row["bon"][32]),
            fmt(row["bon"][64]),
            fmt(row["bon"][128]),
            fmt(row["low128"]),
            fmt(row["oracle128"]),
            f"{row['auc']:.4f}",
            f"{row['corr']:.4f}",
            f"{row['reward_std']:.4f}",
            f"{row['q_range']:.4f}",
        ]
        lines.append("| " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "Notes:",
            "- `BON@N` selects the highest reward among the first N samples for each question.",
            "- `Low@128` selects the lowest reward among all 128 samples; if this beats `BON@128`, reward direction is suspicious.",
            "- `Oracle@128` is pass@128 upper bound on the sampled set.",
            "- `AUC`/`Corr` compare reward with answer correctness over all individual responses.",
            "",
        ]
    )
    return "\n".join(lines)


def render_text(rows):
    headers = [
        "Dataset",
        "Policy",
        "Raw",
        "B@1",
        "B@8",
        "B@16",
        "B@32",
        "B@64",
        "B@128",
        "Low128",
        "Oracle128",
        "AUC",
        "Corr",
    ]
    table = []
    table.append(headers)
    for row in rows:
        table.append(
            [
                row["dataset"],
                row["policy"],
                fmt(row["raw"]),
                fmt(row["bon"][1]),
                fmt(row["bon"][8]),
                fmt(row["bon"][16]),
                fmt(row["bon"][32]),
                fmt(row["bon"][64]),
                fmt(row["bon"][128]),
                fmt(row["low128"]),
                fmt(row["oracle128"]),
                f"{row['auc']:.4f}",
                f"{row['corr']:.4f}",
            ]
        )

    widths = [max(len(str(line[i])) for line in table) for i in range(len(headers))]
    rendered = []
    for idx, line in enumerate(table):
        rendered.append("  ".join(str(value).ljust(widths[i]) for i, value in enumerate(line)))
        if idx == 0:
            rendered.append("  ".join("-" * width for width in widths))
    return "\n".join(rendered) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", required=True)
    parser.add_argument("--math-file", default="data/MATH-500/test.jsonl")
    parser.add_argument("--gsm-file", default="data/GSM-Plus/data/testmini-00000-of-00001.jsonl")
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--output-txt", default=None)
    args = parser.parse_args()

    math_rows = load_original("math", args.math_file)
    gsm_rows = load_original("gsm8k", args.gsm_file)
    originals = {"math": math_rows, "gsm8k": gsm_rows}

    rows = []
    for dataset_name, prefix in (("math", "math"), ("gsm8k", "gsm")):
        for policy in POLICIES:
            scored_file = os.path.join(args.outputs_dir, f"{prefix}-{policy}-scored.json")
            if not os.path.exists(scored_file):
                continue
            rows.append(summarize_one(dataset_name, policy, scored_file, originals[dataset_name]))

    if not rows:
        raise SystemExit(f"No scored files found under {args.outputs_dir}")

    md = render_markdown(rows)
    text = render_text(rows)
    print(text)

    output_md = args.output_md or os.path.join(args.outputs_dir, "eval_summary.md")
    output_txt = args.output_txt or os.path.join(args.outputs_dir, "eval_summary.txt")
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(md)
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Wrote {output_md}")
    print(f"Wrote {output_txt}")


if __name__ == "__main__":
    main()
