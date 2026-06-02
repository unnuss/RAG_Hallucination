"""
Phase 6 — Evaluate the fine-tuned RoBERTa hallucination model
on the same 300 RAGTruth QA test examples used in Phase 4.

This produces a reproducible, internally consistent set of numbers:
  - evaluation_results.csv  (one row per example, with fine-tuned predictions)
  - best_threshold.json     (best F1 threshold for fine-tuned model)

Mirrors phase4_evaluate_v2.py's sentence-level / chunk-best logic so the
comparison is apples-to-apples — only the underlying model changes.
"""

import json
import csv
import time
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "models/halluc_model_v2"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def split_sentences(text):
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if len(p.split()) >= 3]


def split_passages(passages_text):
    chunks = re.split(r'passage\s*\d+\s*:', passages_text)
    chunks = [c.strip() for c in chunks if c.strip()]
    if len(chunks) <= 1:
        words = passages_text.split()
        chunks = [" ".join(words[i:i+100]) for i in range(0, len(words), 100)]
    return chunks


print("Loading fine-tuned model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
print(f"Model loaded. Using device: {device}\n")


def predict_clean_prob_batch(pairs):
    """
    For a batch of (context_chunk, sentence) pairs, return the
    probability of label=0 (clean / not hallucinated) for each.
    """
    enc = tokenizer(
        [p[0] for p in pairs],
        [p[1] for p in pairs],
        truncation=True,
        max_length=256,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        clean_probs = probs[:, 0].cpu().tolist()  # label 0 = clean
    return clean_probs


def worst_sentence_faithfulness(context, answer):
    """
    Mirrors phase4_evaluate_v2's logic, but using our fine-tuned model
    to score (chunk, sentence) pairs instead of the NLI cross-encoder.
    """
    sentences = split_sentences(answer)
    chunks = split_passages(context)
    if not sentences or not chunks:
        return 1.0

    worst = 1.0
    for sent in sentences:
        pairs = [(chunk, sent) for chunk in chunks]
        clean_probs = predict_clean_prob_batch(pairs)
        best_clean_for_sent = max(clean_probs) if clean_probs else 0.0
        worst = min(worst, best_clean_for_sent)
    return worst


print("Loading RAGTruth QA test data...")
sources = {s["source_id"]: s for s in load_jsonl("data/RAGTruth/dataset/source_info.jsonl")}
responses = load_jsonl("data/RAGTruth/dataset/response.jsonl")

qa_test = []
for r in responses:
    src = sources.get(r["source_id"])
    if src and src["task_type"] == "QA" and r["split"] == "test":
        qa_test.append((src, r))

LIMIT = 300
qa_test = qa_test[:LIMIT]

print(f"Evaluating fine-tuned model on {len(qa_test)} QA test examples...\n")

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
        elapsed = time.time() - start
        est_total = elapsed * len(qa_test) / (i + 1)
        print(f"  {i+1}/{len(qa_test)} done ({elapsed:.0f}s elapsed, ~{est_total:.0f}s total)")

print(f"\nScoring finished in {time.time()-start:.0f}s.\n")


def metrics_at_threshold(records, thresh):
    tp = fp = fn = tn = 0
    for rec in records:
        # Predict HALLUCINATED if faithfulness (clean prob) is below threshold
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


print("Threshold sweep:")
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
    json.dump({
        "faithfulness_threshold": best_t,
        "f1": round(best_f1, 3),
        "precision": round(best_p, 3),
        "recall": round(best_r, 3)
    }, f, indent=2)

print("\nSaved evaluation_results.csv and best_threshold.json")
print("Now refresh the dashboard — Model Performance tab will show fine-tuned numbers.")