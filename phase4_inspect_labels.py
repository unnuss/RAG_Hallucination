"""Inspect the structure of RAGTruth hallucination span labels."""
import json

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

sources = {s["source_id"]: s for s in load_jsonl("data/RAGTruth/dataset/source_info.jsonl")}
responses = load_jsonl("data/RAGTruth/dataset/response.jsonl")

# Find a few QA responses WITH labels (hallucinated) in the TRAIN split
shown = 0
for r in responses:
    src = sources.get(r["source_id"])
    if (src and src["task_type"] == "QA"
            and r["split"] == "train" and r["labels"]):
        print("=" * 70)
        print("RESPONSE (first 300 chars):")
        print(r["response"][:300])
        print("\nLABELS (raw):")
        print(json.dumps(r["labels"], indent=2)[:800])
        print("\nFor each label, the text it points to in the response:")
        for lab in r["labels"]:
            if "start" in lab and "end" in lab:
                span_text = r["response"][lab["start"]:lab["end"]]
                print(f"  [{lab.get('label_type', '?')}] -> {span_text!r}")
        shown += 1
        if shown >= 3:
            break

# Also report how many QA train examples we have to work with
qa_train = [r for r in responses
            if sources.get(r["source_id"], {}).get("task_type") == "QA"
            and r["split"] == "train"]
halluc = sum(1 for r in qa_train if r["labels"])
print("\n" + "=" * 70)
print(f"QA train responses: {len(qa_train)}  "
      f"(hallucinated: {halluc}, clean: {len(qa_train) - halluc})")