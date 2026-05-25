"""
Phase 4c — Evaluate our faithfulness scorer on RAGTruth QA test set.

Ground truth: a response is HALLUCINATED if it has any labels, else CLEAN.
Our prediction: we feed (context, answer) to the faithfulness scorer.
  - high faithfulness  -> we predict CLEAN
  - low faithfulness   -> we predict HALLUCINATED

We sweep to find a good threshold, then report precision/recall/F1 and
a confusion matrix, and save per-example results to a CSV for the dashboard.
"""

import json
import csv
import time

from src.faithfulness_scorer import FaithfulnessScorer


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


# ----- Load and filter to QA test split -----
print("Loading data...")
sources = {s["source_id"]: s for s in load_jsonl("data/RAGTruth/dataset/source_info.jsonl")}
responses = load_jsonl("data/RAGTruth/dataset/response.jsonl")

qa_test = []
for r in responses:
    src = sources.get(r["source_id"])
    if src and src["task_type"] == "QA" and r["split"] == "test":
        qa_test.append((src, r))

# Optional: cap the number for a faster first run. Set to None for all 900.
LIMIT = 300
if LIMIT:
    qa_test = qa_test[:LIMIT]

print(f"Evaluating on {len(qa_test)} QA test examples...\n")

# ----- Run our scorer on each example -----
scorer = FaithfulnessScorer()

records = []  # one row per example, for the CSV + metrics
start = time.time()

for i, (src, r) in enumerate(qa_test):
    context = src["source_info"]["passages"]
    answer = r["response"]

    # Ground truth: any labels => hallucinated
    is_hallucinated_truth = 1 if r["labels"] else 0

    # Our model's faithfulness probability (entailment)
    result = scorer.score(context, answer)
    faithfulness = result["faithfulness"]  # 0-1, higher = more faithful

    records.append({
        "id": r["id"],
        "question": src["source_info"]["question"],
        "faithfulness_score": faithfulness,
        "model_verdict": result["verdict"],
        "truth_hallucinated": is_hallucinated_truth,
    })

    if (i + 1) % 25 == 0:
        elapsed = time.time() - start
        print(f"  {i+1}/{len(qa_test)} done ({elapsed:.0f}s)")

print(f"\nScoring finished in {time.time() - start:.0f}s.\n")

# ----- Find the best threshold by sweeping -----
def metrics_at_threshold(records, thresh):
    """Predict hallucinated if faithfulness < thresh. Return tp, fp, fn, tn."""
    tp = fp = fn = tn = 0
    for rec in records:
        pred_halluc = 1 if rec["faithfulness_score"] < thresh else 0
        truth = rec["truth_hallucinated"]
        if pred_halluc and truth:
            tp += 1
        elif pred_halluc and not truth:
            fp += 1
        elif not pred_halluc and truth:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


print("Sweeping thresholds (predict HALLUCINATED when faithfulness < threshold):")
print(f"{'thresh':>7} {'prec':>6} {'recall':>7} {'f1':>6}")
best = None
for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    tp, fp, fn, tn = metrics_at_threshold(records, t)
    p, r_, f1 = prf(tp, fp, fn)
    print(f"{t:>7.1f} {p:>6.2f} {r_:>7.2f} {f1:>6.2f}")
    if best is None or f1 > best[3]:
        best = (t, p, r_, f1)

best_t, best_p, best_r, best_f1 = best
print(f"\nBest threshold = {best_t} (F1 = {best_f1:.2f})")

# ----- Final confusion matrix at best threshold -----
tp, fp, fn, tn = metrics_at_threshold(records, best_t)
print("\nConfusion matrix at best threshold:")
print(f"                 predicted HALLUC   predicted CLEAN")
print(f"  truth HALLUC          {tp:>4}              {fn:>4}")
print(f"  truth CLEAN           {fp:>4}              {tn:>4}")

accuracy = (tp + tn) / len(records)
print(f"\nAccuracy: {accuracy:.2%}")
print(f"Precision (of flagged, how many truly bad): {best_p:.2%}")
print(f"Recall (of truly bad, how many we caught):  {best_r:.2%}")
print(f"F1: {best_f1:.2f}")

# ----- Save per-example results + the chosen threshold to CSV -----
with open("evaluation_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "id", "question", "faithfulness_score", "model_verdict",
        "truth_hallucinated", "predicted_hallucinated"
    ])
    writer.writeheader()
    for rec in records:
        rec_out = dict(rec)
        rec_out["predicted_hallucinated"] = 1 if rec["faithfulness_score"] < best_t else 0
        writer.writerow(rec_out)

# Save the threshold so the dashboard/inspector can reuse it
with open("best_threshold.json", "w") as f:
    json.dump({"faithfulness_threshold": best_t,
               "f1": round(best_f1, 3),
               "precision": round(best_p, 3),
               "recall": round(best_r, 3)}, f, indent=2)

print("\nSaved evaluation_results.csv and best_threshold.json")