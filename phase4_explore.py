"""
Phase 4a — Explore RAGTruth structure.
We just want to SEE what the data looks like before doing anything with it.
"""
import json

# UPDATE these paths to match what `find` printed in Step 1:
SOURCE_PATH = "data/RAGTruth/dataset/source_info.jsonl"
RESPONSE_PATH = "data/RAGTruth/dataset/response.jsonl"


def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


print("Loading source_info...")
sources = load_jsonl(SOURCE_PATH)
print(f"  {len(sources)} source rows")
print("  Keys in a source row:", list(sources[0].keys()))
print("\n  Example source row (trimmed):")
for k, v in sources[0].items():
    preview = str(v)[:200]
    print(f"    {k}: {preview}")

print("\nLoading responses...")
responses = load_jsonl(RESPONSE_PATH)
print(f"  {len(responses)} response rows")
print("  Keys in a response row:", list(responses[0].keys()))
print("\n  Example response row (trimmed):")
for k, v in responses[0].items():
    preview = str(v)[:200]
    print(f"    {k}: {preview}")

# How many responses have hallucinations vs not?
with_halluc = sum(1 for r in responses if r.get("labels"))
print(f"\n  Responses WITH hallucination labels: {with_halluc}")
print(f"  Responses WITHOUT (clean):           {len(responses) - with_halluc}")