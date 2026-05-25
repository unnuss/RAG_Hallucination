"""
Phase 2 — Proof of concept.
Loads our two models and confirms they behave the way our architecture claims:
  - Retrieval model: scores topical relevance between query and context
  - NLI model: scores faithfulness as entailment / contradiction / neutral
This is a throwaway test script. The real organized code comes in Phase 3.
"""

from sentence_transformers import CrossEncoder

# ----------------------------------------------------------------------
# MODEL 1: Retrieval Relevance
# Question it answers: "Is this context relevant to the query?"
# This is a RELEVANCE problem -> we use an MS-MARCO cross-encoder.
# ----------------------------------------------------------------------
print("Loading retrieval relevance model (this may take a moment the first time)...")
retrieval_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ----------------------------------------------------------------------
# MODEL 2: Faithfulness (Natural Language Inference)
# Question it answers: "Does the answer logically follow from the context?"
# This is an NLI problem -> we use a DeBERTa model trained on MNLI/FEVER/ANLI.
# Its 3 outputs map to:  entailment = faithful,
#                        neutral    = unsupported,
#                        contradiction = hallucinated.
# ----------------------------------------------------------------------
print("Loading NLI faithfulness model (this download is larger, please wait)...")
nli_model = CrossEncoder("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")

# The NLI model outputs scores in this label order:
NLI_LABELS = ["entailment", "neutral", "contradiction"]


def softmax(scores):
    """Convert raw model scores into probabilities that add up to 1."""
    import math
    exps = [math.exp(s) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


print("\n" + "=" * 70)
print("TEST 1 — RETRIEVAL RELEVANCE")
print("=" * 70)

query = "What is the company's refund policy?"
relevant_context = "Refunds are processed within 30 days of the purchase date."
irrelevant_context = "Our shipping partners include FedEx, DHL, and UPS."

score_relevant = retrieval_model.predict([(query, relevant_context)])
score_irrelevant = retrieval_model.predict([(query, irrelevant_context)])

print(f"\nQuery: {query}")
print(f"\n  Relevant context:   '{relevant_context}'")
print(f"    -> relevance score: {score_relevant[0]:.3f}  (should be HIGHER)")
print(f"\n  Irrelevant context: '{irrelevant_context}'")
print(f"    -> relevance score: {score_irrelevant[0]:.3f}  (should be LOWER)")


print("\n" + "=" * 70)
print("TEST 2 — FAITHFULNESS (NLI)")
print("=" * 70)

context = "Refunds are processed within 30 days of the purchase date."

faithful_answer = "Customers can receive a refund within 30 days."
hallucinated_answer = "Refunds are available for up to 90 days."
unrelated_answer = "The company offers free shipping on orders over $50."

test_answers = {
    "FAITHFUL  (expect: entailment high)": faithful_answer,
    "HALLUCINATED (expect: contradiction high)": hallucinated_answer,
    "UNRELATED (expect: neutral high)": unrelated_answer,
}

print(f"\nContext: {context}\n")

for label, answer in test_answers.items():
    raw_scores = nli_model.predict([(context, answer)])[0]
    probs = softmax(raw_scores)
    result = dict(zip(NLI_LABELS, probs))

    print(f"  {label}")
    print(f"    Answer: '{answer}'")
    print(f"    entailment={result['entailment']:.2f}  "
          f"neutral={result['neutral']:.2f}  "
          f"contradiction={result['contradiction']:.2f}")
    # Which label won?
    winner = max(result, key=result.get)
    print(f"    -> MODEL SAYS: {winner.upper()}\n")

print("=" * 70)
print("If faithful->entailment, hallucinated->contradiction, "
      "unrelated->neutral,\nthe core architecture works. Phase 2 complete.")
print("=" * 70)