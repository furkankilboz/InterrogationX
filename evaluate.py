"""Evaluation: agreement between two annotators, and the systems against labels.

  python evaluate.py agreement labels/annotator_A.csv labels/annotator_B.csv
  python evaluate.py systems labels/challenge_set.csv
  python evaluate.py systems labels/gold.csv

Systems that are compared:
  majority     always the most frequent label of the file (reference point)
  baseline     similarity + key-point overlap with fixed thresholds
  nli_ungated  NLI without the shared-word filter (to see what the filter changes)
  nli          NLI only
  hybrid       NLI + rule checks + witness uncertainty
  fusion       logistic regression on the scores of all systems, with cross-validation

Rows without a label still get predictions but are not counted in the metrics. Results go to results/.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import config
from nli_consistency import analyze_consistency
from statement_similarity_analysis import analyze_statements

LABELS = list(config.LABELS)
SYSTEMS = ["baseline", "nli_ungated", "nli", "hybrid"]
FEATURES = ["similarity", "overlap", "contradiction", "contradiction_ungated", "support_ratio",
            "polarity_conflicts", "time_conflicts", "number_conflicts", "hedges"]


def read_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    invalid = []
    for row in rows:
        label = row.get("label", "").strip()
        if label and label not in LABELS:
            invalid.append(row["pair_id"])
    if invalid:
        raise SystemExit(f"{path}: invalid labels for {invalid}; use one of {', '.join(LABELS)}")
    return rows


def read_labels(path):
    """pair_id -> label, only for rows that have a label."""
    labels = {}
    for row in read_rows(path):
        if row["label"].strip():
            labels[row["pair_id"]] = row["label"].strip()
    return labels


def metrics(gold, predicted):
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(gold, predicted, labels=LABELS, zero_division=0)
    per_class = {}
    for i, label in enumerate(LABELS):
        per_class[label] = {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]),
                            "support": int(support[i])}
    return {
        "accuracy": accuracy_score(gold, predicted),
        "macro_f1": float(f1.mean()),
        "per_class": per_class,
        "confusion_matrix": {"labels": LABELS,
                             "rows_gold_columns_predicted": confusion_matrix(gold, predicted, labels=LABELS).tolist()},
    }


def agreement(path_a, path_b):
    from sklearn.metrics import cohen_kappa_score, confusion_matrix

    a = read_labels(path_a)
    b = read_labels(path_b)
    common = sorted(set(a) & set(b))
    if not common:
        raise SystemExit("No pair is labeled in both files.")
    labels_a = [a[pair] for pair in common]
    labels_b = [b[pair] for pair in common]

    same = 0
    for label_a, label_b in zip(labels_a, labels_b):
        if label_a == label_b:
            same += 1
    print(f"Pairs labeled by both: {len(common)}")
    print(f"Raw agreement: {same / len(common):.2f}")
    print(f"Cohen's kappa: {cohen_kappa_score(labels_a, labels_b, labels=LABELS):.2f}")
    print(f"Confusion (rows {Path(path_a).name}, columns {Path(path_b).name}), order {LABELS}:")
    for row in confusion_matrix(labels_a, labels_b, labels=LABELS):
        print("  ", list(row))

    if same < len(common):
        print("Disagreements to discuss:")
        for pair in common:
            if a[pair] != b[pair]:
                print(f"  {pair}: {a[pair]} vs {b[pair]}")


def analyze_row(row):
    """Predictions of every system and the scores they are based on, for one pair."""
    base = analyze_statements(row["suspect_statement"], row["witness_statement"])
    consistency = analyze_consistency(row["suspect_statement"], row["witness_statement"])
    return {
        "baseline": base["verdict"],
        "nli_ungated": consistency["nli_ungated_verdict"],
        "nli": consistency["nli_verdict"],
        "hybrid": consistency["verdict"],
        "similarity": base["similarity"],
        "overlap": base["overlap"],
        "contradiction": consistency["contradiction"],
        "contradiction_ungated": consistency["contradiction_ungated"],
        "support_ratio": consistency["support_ratio"],
        "polarity_conflicts": len(consistency["rule_conflicts"]["polarity"]),
        "time_conflicts": len(consistency["rule_conflicts"]["time"]),
        "number_conflicts": len(consistency["rule_conflicts"]["number"]),
        "hedges": len(consistency["hedges"]),
    }


def fusion_predictions(analyses, gold, seed=0):
    """Train a small classifier on the scores. Every pair is predicted by a model that did not see its label
    (cross-validation). Returns (predictions, number of folds), or (None, None) with too few labels."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    smallest_class = min(Counter(gold).values())
    folds = min(5, smallest_class)
    if folds < 2 or len(set(gold)) < 2:
        return None, None
    X = []
    for analysis in analyses:
        X.append([analysis[name] for name in FEATURES])
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    splitter = StratifiedKFold(folds, shuffle=True, random_state=seed)
    return list(cross_val_predict(model, X, gold, cv=splitter)), folds


def accuracy_by_category(rows, gold, predicted):
    correct = Counter()
    total = Counter()
    for row, gold_label, predicted_label in zip(rows, gold, predicted):
        total[row["category"]] += 1
        if gold_label == predicted_label:
            correct[row["category"]] += 1
    return {category: correct[category] / total[category] for category in total}


def evaluate_systems(path):
    rows = read_rows(path)
    print(f"Analyzing {len(rows)} pairs from {path} ...")
    analyses = []
    for row in rows:
        analysis = analyze_row(row)
        analyses.append(analysis)
        for system in SYSTEMS:
            row[system] = analysis[system]
        for name in FEATURES:
            row[name] = round(analysis[name], 4)

    labeled_rows = []
    labeled_analyses = []
    for row, analysis in zip(rows, analyses):
        if row["label"].strip():
            labeled_rows.append(row)
            labeled_analyses.append(analysis)

    report = {"file": str(path), "pairs": len(rows), "labeled_pairs": len(labeled_rows), "systems": {}}
    if labeled_rows:
        gold = [row["label"].strip() for row in labeled_rows]
        majority_label = Counter(gold).most_common(1)[0][0]
        predictions = {"majority": [majority_label] * len(gold)}
        for system in SYSTEMS:
            predictions[system] = [row[system] for row in labeled_rows]
        fused, folds = fusion_predictions(labeled_analyses, gold)
        if fused:
            predictions["fusion"] = fused
            report["fusion_folds"] = folds
            for row, label in zip(labeled_rows, fused):
                row["fusion"] = label

        for system, predicted in predictions.items():
            report["systems"][system] = metrics(gold, predicted)
            if "category" in labeled_rows[0]:
                report["systems"][system]["accuracy_by_category"] = accuracy_by_category(labeled_rows, gold,
                                                                                         predicted)
        print_report(report, gold)
    else:
        print("No labels in this file yet: wrote predictions only.")
        for system in SYSTEMS:
            print(f"  {system:<13} {dict(Counter(row[system] for row in rows))}")

    save_results(path, rows, report)


def save_results(path, rows, report):
    config.RESULTS_DIR.mkdir(exist_ok=True)
    name = Path(path).stem
    columns = list(rows[0].keys())
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with open(config.RESULTS_DIR / f"{name}_predictions.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    with open(config.RESULTS_DIR / f"{name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved {config.RESULTS_DIR / f'{name}_predictions.csv'} and {name}_metrics.json")


def print_report(report, gold):
    systems = report["systems"]
    print(f"\nLabeled pairs: {report['labeled_pairs']}  ({dict(Counter(gold))})")
    print(f"\n{'system':<13}{'accuracy':>10}{'macro-F1':>10}   per-class F1 ({', '.join(LABELS)})")
    for system, result in systems.items():
        per_class = "  ".join(f"{result['per_class'][label]['f1']:.2f}" for label in LABELS)
        print(f"{system:<13}{result['accuracy']:>10.2f}{result['macro_f1']:>10.2f}   {per_class}")
    if "fusion_folds" in report:
        print(f"(fusion: {report['fusion_folds']}-fold cross-validated predictions)")

    first_system = next(iter(systems.values()))
    if "accuracy_by_category" in first_system:
        print("\nAccuracy by category")
        print(f"{'category':<22}" + "".join(f"{system:>13}" for system in systems))
        for category in first_system["accuracy_by_category"]:
            line = f"{category:<22}"
            for result in systems.values():
                line += f"{result['accuracy_by_category'][category]:>13.2f}"
            print(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    agreement_parser = commands.add_parser("agreement", help="Cohen's kappa between two annotator files")
    agreement_parser.add_argument("file_a")
    agreement_parser.add_argument("file_b")
    systems_parser = commands.add_parser("systems", help="evaluate all systems on a labeled CSV")
    systems_parser.add_argument("labels", nargs="?", default=str(config.CHALLENGE_SET))
    args = parser.parse_args()
    if args.command == "agreement":
        agreement(args.file_a, args.file_b)
    else:
        evaluate_systems(args.labels)


if __name__ == "__main__":
    main()
