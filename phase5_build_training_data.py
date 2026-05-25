"""
Build sentence-level training data for fine-tuning (Option A).

For each QA training response:
  - split the answer into sentences (tracking each sentence's char position)
  - a sentence is HALLUCINATED (label 1) if it overlaps any labeled span,
    otherwise CLEAN (label 0)
  - pair each sentence with the most relevant context chunk

Output: a JSONL of {context_chunk, sentence, label} rows, ready for Colab.
"""

import json
import re

from src.faithfulness_scorer import FaithfulnessScorer, NLI_LABELS, _softmax


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def split_sentences_with_spans(text):
    """Split into sentences, returning (sentence, start_char, end_char)."""
    results = []
    for m in re.finditer(r'[^.!?]*[.!?]', text):
        sent = m.group().strip()
        if len(sent.split()) >= 3:
            results.append((sent, m.start(), m.end()))
    # Catch trailing text with no final punctuation
    if not text.rstrip().endswith((".", "!", "?")):
        tail_start = text.rfind(results[-1][0]) + len(results[-1][0]) if results else 0
        tail = text[tail_start:].strip()
        if len(tail.split()) >= 3:
            results.append((tail, tail_start, len(text)))
    return results


def split_passages(passages_text):
    chunks = re.split(r'passage\s*\d+\s*:', passages_text)
    chunks = [c.strip() for c in chunks if c.strip()]
    if len(chunks) <= 1:
        words = passages_text.split()
        chunks = [" ".join(words[i:i+100]) for i in range(0, len(words), 100)]
    return chunks


def spans_overlap(s1, e1, s2, e2):
    return s1 < e2 and s2 < e1


print("Loading RAGTruth...")
sources = {s["source_id"]: s for s in load_jsonl("data/RAGTruth/dataset/source_info.jsonl")}
responses = load_jsonl("data/RAGTruth/dataset/response.jsonl")

qa_train = [(sources[r["source_id"]], r) for r in responses
            if sources.get(r["source_id"], {}).get("task_type") == "QA"
            and r["split"] == "train"]

print(f"Processing {len(qa_train)} QA training responses...")

# We use the model only to pick the best-matching chunk for each sentence.
scorer = FaithfulnessScorer()
model = scorer.model


def best_chunk_for_sentence(sentence, chunks):
    """Return the chunk that most entails this sentence (best context match)."""
    pairs = [(c, sentence) for c in chunks]
    raw = model.predict(pairs)
    best_chunk, best_entail = chunks[0], -1.0
    for chunk, r_ in zip(chunks, raw):
        entail = dict(zip(NLI_LABELS, _softmax(r_)))["entailment"]
        if entail > best_entail:
            best_entail, best_chunk = entail, chunk
    return best_chunk


training_rows = []
for idx, (src, r) in enumerate(qa_train):
    response_text = r["response"]
    passages = src["source_info"]["passages"]
    chunks = split_passages(passages)
    if not chunks:
        continue

    halluc_spans = [(lab["start"], lab["end"]) for lab in r["labels"]
                    if "start" in lab and "end" in lab]

    for sent, s_start, s_end in split_sentences_with_spans(response_text):
        # Sentence is hallucinated if it overlaps any labeled span
        is_halluc = any(spans_overlap(s_start, s_end, hs, he)
                        for hs, he in halluc_spans)
        chunk = best_chunk_for_sentence(sent, chunks)
        training_rows.append({
            "context": chunk,
            "sentence": sent,
            "label": 1 if is_halluc else 0,  # 1 = hallucinated, 0 = clean
        })

    if (idx + 1) % 200 == 0:
        print(f"  {idx+1}/{len(qa_train)} responses processed, "
              f"{len(training_rows)} sentence pairs so far")

# Report balance
n_halluc = sum(1 for row in training_rows if row["label"] == 1)
print(f"\nTotal sentence pairs: {len(training_rows)}")
print(f"  Hallucinated: {n_halluc}")
print(f"  Clean:        {len(training_rows) - n_halluc}")

with open("training_data.jsonl", "w") as f:
    for row in training_rows:
        f.write(json.dumps(row) + "\n")

print("\nSaved training_data.jsonl — upload this to Colab next.")