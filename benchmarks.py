"""Tests on data from outside the project. The data is downloaded when needed (see external_datasets.py) and
the results are saved to results/.

  python benchmarks.py dnli  [--models A B ...] [--pairs 600]     contradiction detection on Dialogue NLI
  python benchmarks.py alibi [--models A B ...] [--sessions 80]   lie/truth alibis in interrogation sessions
  python benchmarks.py cues                                       why body language is not scored

Every benchmark takes a fixed random sample and splits it in two halves. The "dev" half is for choosing a
model or a threshold, the "test" half is only for the final numbers.

dnli   Sentence pairs checked by humans (entailment / neutral / contradiction).
alibi  Synthetic interrogations where the character card says if the alibi is a lie. A session should conflict
       with the true facts when the alibi is a lie. Two session scores are compared: the evidence conflict
       index and the single strongest contradiction.
cues   A classifier separates "culprit" from "genuine" rows of a synthetic dataset almost perfectly from the
       body language words alone. This shows that the dataset is built on a stereotype. It does not show that
       body language reveals lies.
"""
import argparse
import json
import random
import time

import config
import models
from external_datasets import CUES, DNLI, download, iter_alibi_sessions, read_jsonl

SEED = 13
DNLI_LABELS = {0: "entailment", 1: "neutral", 2: "contradiction"}
SIMILARITY_NAME = "e5 similarity (low = contradiction)"


def split_dev_test(items):
    """Two halves of an already shuffled list: even positions are dev, odd positions are test."""
    return {"dev": items[0::2], "test": items[1::2]}


def nli_scores(pairs, model_name):
    model = models.get_nli_model(model_name)
    labels = [model.config.id2label[i].lower() for i in range(len(model.config.id2label))]
    scores = model.predict(pairs, apply_softmax=True, batch_size=32, show_progress_bar=False)
    results = []
    for row in scores:
        results.append({label: float(score) for label, score in zip(labels, row)})
    return results


def similarity_scores(pairs):
    model = models.get_similarity_model()
    left = model.encode([config.SIMILARITY_PREFIX + a for a, b in pairs], normalize_embeddings=True)
    right = model.encode([config.SIMILARITY_PREFIX + b for a, b in pairs], normalize_embeddings=True)
    return [float(x @ y) for x, y in zip(left, right)]


def binary_metrics(gold, scores, threshold):
    """gold: list of True/False. scores: a higher score means "more likely True"."""
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

    predicted = [score >= threshold for score in scores]
    precision, recall, f1, _ = precision_recall_fscore_support(gold, predicted, average="binary", zero_division=0)
    return {"auc": float(roc_auc_score(gold, scores)), "precision": float(precision), "recall": float(recall),
            "f1": float(f1), "threshold": threshold}


def best_threshold(gold, scores):
    """The threshold with the best F1 on the given data. With equal F1 the lower threshold wins."""
    best = None
    best_f1 = -1.0
    for step in range(-19, 20):
        threshold = round(0.05 * step, 2)
        f1 = binary_metrics(gold, scores, threshold)["f1"]
        if f1 > best_f1:
            best = threshold
            best_f1 = f1
    return best


def add_dev_tuned_result(system, gold, scores):
    """Pick the threshold on dev and use it unchanged on test. The raw scores are saved too."""
    threshold = best_threshold(gold["dev"], scores["dev"])
    system["dev_tuned_threshold"] = threshold
    system["test_at_dev_tuned_threshold"] = binary_metrics(gold["test"], scores["test"], threshold)
    system["scores"] = {}
    for split, values in scores.items():
        system["scores"][split] = [round(value, 4) for value in values]


def similarity_baseline(gold, text_pairs):
    """AUC when a low e5 similarity is read as "contradiction"."""
    result = {}
    for split in gold:
        scores = [-score for score in similarity_scores(text_pairs[split])]
        result[split] = {"auc": binary_metrics(gold[split], scores, 0.0)["auc"]}
    return result


def load_dnli_sample(pairs):
    """The same number of pairs from each of the three classes, shuffled."""
    import pandas as pd

    frame = pd.read_parquet(download(DNLI))
    rows = []
    for label_id, label in DNLI_LABELS.items():
        subset = frame[frame.label == label_id].sample(pairs // 3, random_state=SEED)
        for row in subset.itertuples():
            rows.append((row.sentence1, row.sentence2, label))
    random.Random(SEED).shuffle(rows)
    return rows


def run_dnli(model_names, pairs):
    splits = split_dev_test(load_dnli_sample(pairs))
    gold = {}
    text_pairs = {}
    for split, rows in splits.items():
        gold[split] = [label == "contradiction" for a, b, label in rows]
        text_pairs[split] = [(a, b) for a, b, label in rows]

    report = {"benchmark": "dnli", "source": DNLI[0], "pairs_per_split": {"dev": len(splits["dev"]),
                                                                          "test": len(splits["test"])},
              "gold": gold, "systems": {}}
    report["systems"][SIMILARITY_NAME] = similarity_baseline(gold, text_pairs)

    for name in model_names:
        system = {}
        scores = {}
        for split, rows in splits.items():
            start = time.time()
            probabilities = nli_scores(text_pairs[split], name)
            seconds = time.time() - start
            scores[split] = [p["contradiction"] for p in probabilities]

            correct = 0
            for p, (a, b, label) in zip(probabilities, rows):
                if max(p, key=p.get) == label:
                    correct += 1
            system[split] = binary_metrics(gold[split], scores[split], config.NLI_STRONG_CONTRADICTION)
            system[split]["three_way_accuracy"] = correct / len(rows)
            system[split]["pairs_per_second"] = len(rows) / seconds
        add_dev_tuned_result(system, gold, scores)
        report["systems"][name] = system
    return report


def load_alibi_sample(sessions, min_turns=4, max_turns=8):
    """Half sessions with a false alibi and half with a true one, shuffled."""
    lies = []
    truths = []
    for session in iter_alibi_sessions():
        if min_turns <= len(session["turns"]) <= max_turns:
            if session["ground_truth"]["alibi"] == "lie":
                lies.append(session)
            else:
                truths.append(session)
    rng = random.Random(SEED)
    sample = rng.sample(lies, sessions // 2) + rng.sample(truths, sessions // 2)
    rng.shuffle(sample)
    return sample


def run_alibi(model_names, sessions):
    from interrogation_analysis import score_session_against_evidence

    splits = split_dev_test(load_alibi_sample(sessions))
    gold = {}
    text_pairs = {}
    for split, items in splits.items():
        gold[split] = [session["ground_truth"]["alibi"] == "lie" for session in items]
        text_pairs[split] = []
        for session in items:
            answers = " ".join(turn["answer"] for turn in session["turns"])
            facts = " ".join(item["text"] for item in session["evidence"])
            text_pairs[split].append((answers, facts))

    report = {"benchmark": "alibi", "source": "solarkyle/alibi-interrogation-dataset",
              "sessions_per_split": {"dev": len(splits["dev"]), "test": len(splits["test"])},
              "gold": gold, "systems": {}}
    report["systems"][SIMILARITY_NAME] = similarity_baseline(gold, text_pairs)

    # Fixed thresholds. An index above 0 means the account conflicts with more evidence than it acknowledges.
    fixed_thresholds = {"conflict index": 0.0, "strongest contradiction": config.SESSION_CONTRADICTION_THRESHOLD}
    configured_model = config.NLI_MODEL
    for name in model_names:
        config.NLI_MODEL = name
        results = {}
        seconds = {}
        for split, items in splits.items():
            start = time.time()
            results[split] = [score_session_against_evidence(session) for session in items]
            seconds[split] = (time.time() - start) / len(items)
            print(f"  done: {name} on {split} ({len(items)} sessions)", flush=True)

        for score_name, threshold in fixed_thresholds.items():
            scores = {}
            system = {}
            for split in splits:
                scores[split] = [result[score_name] for result in results[split]]
                system[split] = binary_metrics(gold[split], scores[split], threshold)
                system[split]["seconds_per_session"] = seconds[split]
            add_dev_tuned_result(system, gold, scores)
            report["systems"][f"{name} | {score_name}"] = system
    config.NLI_MODEL = configured_model
    return report


def run_cues():
    """Can a simple classifier tell "culprit" from "genuine" rows, and from which part of the row?"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline

    rows = read_jsonl(download(CUES))
    gold = [row["tag"] for row in rows]
    views = {
        "nonverbal cues only": [" ".join((row.get("cues") or {}).values()) for row in rows],
        "answer text only": [row["answer"] for row in rows],
        "question text only": [row["question"] for row in rows],
    }
    majority = max(gold.count(tag) for tag in set(gold)) / len(gold)
    report = {"benchmark": "cues", "source": CUES[0], "rows": len(rows), "majority_accuracy": majority, "views": {}}
    folds = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for name, texts in views.items():
        classifier = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
        report["views"][name] = float(cross_val_score(classifier, texts, gold, cv=folds).mean())
    return report


def print_report(report):
    print(f"\n== {report['benchmark']} ({report['source']})")
    if report["benchmark"] == "cues":
        print(f"rows: {report['rows']}, majority class accuracy: {report['majority_accuracy']:.3f}")
        for view, accuracy in report["views"].items():
            print(f"  5-fold accuracy from {view:<22} {accuracy:.3f}")
        return

    for system, results in report["systems"].items():
        for split in ("dev", "test"):
            numbers = []
            for key, value in results[split].items():
                if key != "threshold":
                    numbers.append(f"{key}={value:.3f}")
            print(f"  {system:<58} {split:<5} " + "  ".join(numbers))
        if "dev_tuned_threshold" in results:
            tuned = results["test_at_dev_tuned_threshold"]
            print(f"  {'':<58} test at dev-tuned threshold {results['dev_tuned_threshold']:.2f}: "
                  f"precision={tuned['precision']:.3f}  recall={tuned['recall']:.3f}  f1={tuned['f1']:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("benchmark", choices=["dnli", "alibi", "cues"])
    parser.add_argument("--models", nargs="+", default=[config.NLI_MODEL])
    parser.add_argument("--pairs", type=int, default=600)
    parser.add_argument("--sessions", type=int, default=80)
    parser.add_argument("--tag", default="", help="added to the name of the result file")
    args = parser.parse_args()

    if args.benchmark == "dnli":
        report = run_dnli(args.models, args.pairs)
    elif args.benchmark == "alibi":
        report = run_alibi(args.models, args.sessions)
    else:
        report = run_cues()
    print_report(report)

    config.RESULTS_DIR.mkdir(exist_ok=True)
    path = config.RESULTS_DIR / f"benchmark_{args.benchmark}{args.tag}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
