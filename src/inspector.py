"""
The Inspector — the core of the RAG Reliability product.

Runs both stages (retrieval relevance + generation faithfulness) and
combines them into a single diagnostic verdict that tells a developer
NOT JUST that something is wrong, but WHICH part of their pipeline failed.

Faithfulness is now powered by our FINE-TUNED model (RoBERTa fine-tuned
on RAGTruth + contradiction examples), which detects baseless info,
contradictions, and unsupported content.
"""

from dataclasses import dataclass, asdict
from typing import List, Union

from src.retrieval_scorer import RetrievalScorer
from src.faithfulness_scorer import FaithfulnessScorer


@dataclass
class InspectionResult:
    """Structured output of one reliability inspection."""
    retrieval_relevance: float
    faithfulness_score: float
    hallucination_score: float
    faithfulness_verdict: str   # FAITHFUL / HALLUCINATED
    overall_status: str         # RELIABLE / LOW / VERY_LOW
    diagnosis: str              # human-readable explanation

    def to_dict(self) -> dict:
        return asdict(self)


class Inspector:
    """Two-stage RAG reliability inspector."""

    # Retrieval threshold calibrated to the MS-MARCO score scale (Phase 3).
    RETRIEVAL_THRESHOLD = 0.05
    # Faithfulness threshold: flag if P(clean) falls below this.
    FAITHFULNESS_THRESHOLD = 0.5

    def __init__(self):
        print("Loading models (first run may take a moment)...")
        self.retrieval_scorer = RetrievalScorer()
        self.faithfulness_scorer = FaithfulnessScorer()
        print("Inspector ready.\n")

    def evaluate(
        self,
        query: str,
        context: Union[str, List[str]],
        answer: str,
    ) -> InspectionResult:
        # Stage 1: retrieval relevance (query vs context)
        retrieval_score = self.retrieval_scorer.score(query, context)

        # Stage 2: faithfulness (context vs answer)
        context_text = " ".join(context) if isinstance(context, list) else context
        faith = self.faithfulness_scorer.score(context_text, answer)

        retrieval_ok = retrieval_score >= self.RETRIEVAL_THRESHOLD
        faithful_ok = faith["faithfulness"] >= self.FAITHFULNESS_THRESHOLD

        # The diagnosis logic — the product's core value.
        if retrieval_ok and faithful_ok:
            status = "RELIABLE"
            diagnosis = "Retrieval and generation both look healthy."
        elif not retrieval_ok and faithful_ok:
            status = "LOW"
            diagnosis = (
                "Answer is internally faithful, but to weakly relevant context. "
                "The retriever likely fetched the wrong documents — "
                "investigate embeddings, chunking, or the knowledge base."
            )
        elif retrieval_ok and not faithful_ok:
            status = "LOW"
            diagnosis = (
                "Correct context was retrieved, but the answer is NOT grounded in it. "
                "This is a generation hallucination — the model produced unsupported "
                "or contradictory information. Check prompt or model settings."
            )
        else:
            status = "VERY_LOW"
            diagnosis = (
                "Both retrieval and faithfulness are weak — "
                "possible cascading failure across the pipeline."
            )

        return InspectionResult(
            retrieval_relevance=round(retrieval_score, 3),
            faithfulness_score=faith["faithfulness"],
            hallucination_score=faith["hallucination"],
            faithfulness_verdict=faith["verdict"],
            overall_status=status,
            diagnosis=diagnosis,
        )


if __name__ == "__main__":
    inspector = Inspector()

    cases = [
        {
            "name": "Healthy",
            "query": "What is the refund policy?",
            "context": "Refunds are processed within 30 days of purchase.",
            "answer": "Customers can get a refund within 30 days.",
        },
        {
            "name": "Generation hallucination (contradiction)",
            "query": "What is the refund policy?",
            "context": "Refunds are processed within 30 days of purchase.",
            "answer": "Refunds are available for up to 90 days.",
        },
        {
            "name": "Retrieval failure",
            "query": "What is the refund policy?",
            "context": "Our shipping partners include FedEx, DHL, and UPS.",
            "answer": "The company ships via FedEx, DHL, and UPS.",
        },
        {
            "name": "Cascading failure",
            "query": "What is the refund policy?",
            "context": "The office is open Monday to Friday, 9am to 5pm.",
            "answer": "Refunds must be requested within 7 days with a receipt.",
        },
    ]

    for c in cases:
        print("=" * 70)
        print(f"CASE: {c['name']}")
        result = inspector.evaluate(c["query"], c["context"], c["answer"])
        print(f"  Retrieval relevance: {result.retrieval_relevance}")
        print(f"  Faithfulness:        {result.faithfulness_score} "
              f"({result.faithfulness_verdict})")
        print(f"  Hallucination score: {result.hallucination_score}")
        print(f"  Status:              {result.overall_status}")
        print(f"  Diagnosis:           {result.diagnosis}")
    print("=" * 70)