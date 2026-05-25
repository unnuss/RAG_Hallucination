"""
Phase 4d — Sentence-level faithfulness evaluation.

Instead of one (whole context, whole answer) comparison, we:
  1. split the answer into sentences
  2. split the context into passage chunks
  3. for each answer sentence, find its best-matching context chunk
     and run NLI on that short pair
  4. flag the answer as hallucinated if its WORST sentence looks
     contradicted/unsupported

This respects the NLI model's context window and matches RAGTruth's
span-level nature. It also lays the groundwork for span-level attribution.
"""

import json
import csv
import time
import re

from src.faithfulness_scorer import FaithfulnessScorer, NLI_LABELS, _softmax


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def split_sentences(text):
    # Simple sentence splitter — good enough for evaluation.
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if len(p.split()) >= 3]


def split_passages(passages_text):
    # RAGTruth passages are separated by "passage N:" markers.
    chunks = re.split(r'passage\s*\d+\s*:', passages_text)
    chunks = [c.strip() for c in chunks if c.strip()]
    # Fallback: if no markers, chunk by ~100 words.
    if len(chunks) <= 1:
        words = passages_text.split()
        chunks = [" ".join(words[i:i+100]) for i in range(0, len(words), 100)]
    return chunks


print("Loading data...")
sources = {s["source_id"]: s for s in load_jsonl("data/RAGTruth/dataset/source_info.jsonl")}
responses = load_jsonl("data/RAGTruth/dataset/response.jsonl")

qa_test = []
for r in responses:
    src = sources.get(r["source_id"])
    if src and src["task_type"] == "QA" and r["split"] == "test":
        qa_test.append((src, r))

LIMIT = 300
if LIMIT:
    qa_test = qa_test[:LIMIT]

print(f"Evaluating on {len(qa_test)} QA test examples (sentence-level)...\n")

scorer = FaithfulnessScorer()
model = scorer.model  # reuse the underlying CrossEncoder for batch scoring


def worst_sentence_faithfulness(context, answer):
    """
    Return the LOWEST faithfulness score across answer sentences.
    Low = at least one sentence isn't supported = likely hallucination.
    """
    sentences = split_sentences(answer)
    chunks = split_passages(context)
    if not sentences or not chunks:
        return 1.0  # nothing to flag

    worst = 1.0
    for sent in sentences:
        # Build (chunk, sentence) pairs; pick the chunk that BEST supports it.
        pairs = [(chunk, sent) for chunk in chunks]
        raw = model.predict(pairs)  # batch over chunks
        # For each chunk, entailment probability for this sentence
        best_entail_for_sent = 0.0
        for r_ in raw:
            probs = _softmax(r_)
            entail = dict(zip(NLI_LABELS, probs))["entailment"]
            best_entail_for_sent = max(best_entail_for_sent, entail)
        # A sentence is "supported" if its best chunk entails it.
        worst = min(worst, best_entail_for_sent)
    return worst


records = []
start = time.time()

for i, (src, r) in enumerate(qa_test):
    context = src["source_info"]["passages"]
    answer = r["response"]
    truth = 1 if r["labels"] else 0

    score = worst_sentence_faithfulness(context, answer)

    records.append({
        "id": r["id"],
        "question": src["source_info"]["question"],
        "faithfulness_score": round(float(score), 4),
        "truth_hallucinated": truth,
    })

    if (i + 1) % 25 == 0:
        print(f"  {i+1}/{len(qa_test)} done ({time.time()-start:.0f}s)")

print(f"\nScoring finished in {time.time()-start:.0f}s.\n")


def metrics_at_threshold(records, thresh):
    tp = fp = fn = tn = 0
    for rec in records:
        pred = 1 if rec["faithfulness_score"] < thresh else 0
        truth = rec["truth_hallucinated"]
        if pred and truth: tp += 1
        elif pred and not truth: fp += 1
        elif not pred and truth: fn += 1
        else: tn += 1
    return tp, fp, fn, tn


def prf(tp, fp, fn):
    p = tp/(tp+fp) if (tp+fp) else 0.0
    r = tp/(tp+fn) if (tp+fn) else 0.0
    f1 = 2*p*r/(p+r) if (p+r) else 0.0
    return p, r, f1


print("Threshold sweep (predict HALLUCINATED when worst-sentence score < threshold):")
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

tp, fp, fn, tn = metrics_at_threshold(records, best_t)
print("\nConfusion matrix at best threshold:")
print(f"                 predicted HALLUC   predicted CLEAN")
print(f"  truth HALLUC          {tp:>4}              {fn:>4}")
print(f"  truth CLEAN           {fp:>4}              {tn:>4}")
print(f"\nAccuracy:  {(tp+tn)/len(records):.2%}")
print(f"Precision: {best_p:.2%}")
print(f"Recall:    {best_r:.2%}")
print(f"F1:        {best_f1:.2f}")

with open("evaluation_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=[
        "id", "question", "faithfulness_score",
        "truth_hallucinated", "predicted_hallucinated"])
    w.writeheader()
    for rec in records:
        out = dict(rec)
        out["predicted_hallucinated"] = 1 if rec["faithfulness_score"] < best_t else 0
        w.writerow(out)

with open("best_threshold.json", "w") as f:
    json.dump({"faithfulness_threshold": best_t, "f1": round(best_f1,3),
               "precision": round(best_p,3), "recall": round(best_r,3)}, f, indent=2)

print("\nSaved evaluation_results.csv and best_threshold.json")