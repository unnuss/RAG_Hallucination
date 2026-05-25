"""
Retrieval Relevance Scorer.

Wraps an MS-MARCO cross-encoder to answer: "Is the retrieved context
relevant to the user's query?" This is a RELEVANCE problem.

The raw model outputs logits (can be negative), so we squash them into
a clean 0-1 range with a sigmoid for nice display and thresholding.
"""

import math
from typing import List, Union
from sentence_transformers import CrossEncoder


def _sigmoid(x: float) -> float:
    """Squash any real number into the range (0, 1)."""
    return 1.0 / (1.0 + math.exp(-x))


class RetrievalScorer:
    """Scores how relevant retrieved context is to a query."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def score(self, query: str, context: Union[str, List[str]]) -> float:
        """
        Args:
            query: the user's question.
            context: a single context string, or a list of retrieved chunks.
                     If a list, we score each chunk and take the best one
                     (a query is well-served if ANY chunk is relevant).

        Returns:
            A relevance score in [0, 1]. Higher = more relevant.
        """
        if isinstance(context, str):
            context = [context]

        pairs = [(query, chunk) for chunk in context]
        raw_scores = self.model.predict(pairs)

        # Squash each logit to 0-1, then take the best-scoring chunk.
        normalized = [_sigmoid(float(s)) for s in raw_scores]
        return max(normalized)


# Quick self-test: run this file directly to sanity-check it.
if __name__ == "__main__":
    scorer = RetrievalScorer()
    q = "What is the company's refund policy?"
    print("Relevant:  ", round(scorer.score(q, "Refunds are processed within 30 days."), 3))
    print("Irrelevant:", round(scorer.score(q, "Our shipping partners include FedEx and DHL."), 3))