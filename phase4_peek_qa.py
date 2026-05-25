"""Phase 4b — look at one QA example to confirm context structure."""
import json

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

sources = {s["source_id"]: s for s in load_jsonl("data/RAGTruth/dataset/source_info.jsonl")}
responses = load_jsonl("data/RAGTruth/dataset/response.jsonl")

# Find first QA response in the test split
for r in responses:
    src = sources.get(r["source_id"])
    if src and src["task_type"] == "QA" and r["split"] == "test":
        print("TASK TYPE:", src["task_type"])
        print("\nSOURCE_INFO type:", type(src["source_info"]).__name__)
        print("\nSOURCE_INFO (first 800 chars):")
        print(str(src["source_info"])[:800])
        print("\nRESPONSE (first 400 chars):")
        print(r["response"][:400])
        print("\nLABELS:", r["labels"][:2])
        break

# Count QA test examples available
qa_test = [r for r in responses
           if sources.get(r["source_id"], {}).get("task_type") == "QA"
           and r["split"] == "test"]
print(f"\nTotal QA test responses available: {len(qa_test)}")
halluc = sum(1 for r in qa_test if r["labels"])
print(f"  Hallucinated: {halluc}   Clean: {len(qa_test) - halluc}")