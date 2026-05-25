"""
Generation Faithfulness Scorer — now using our FINE-TUNED model.

Our model is a 2-class classifier trained on RAGTruth (+ contradiction
examples): given (context, sentence), it predicts:
    label 0 = clean (faithful / supported)
    label 1 = hallucinated (unsupported or contradictory)

We expose a 'faithfulness' score = P(clean), so higher = more faithful,
keeping the same interface the rest of the project already expects.
"""

from typing import Dict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class FaithfulnessScorer:
    """Scores whether an answer sentence is grounded in its context."""

    def __init__(self, model_path: str = "models/halluc_model_v2"):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

    def score(self, context: str, answer: str) -> Dict[str, float]:
        """
        Args:
            context: the retrieved context (or context chunk).
            answer:  the generated answer (or answer sentence).

        Returns:
            dict with:
                faithfulness  -> P(clean), 0-1, higher = more faithful
                hallucination -> P(hallucinated)
                verdict       -> "FAITHFUL" or "HALLUCINATED"
        """
        inputs = self.tokenizer(
            context, answer, truncation=True, max_length=256,
            return_tensors="pt"
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0]

        p_clean = float(probs[0])
        p_halluc = float(probs[1])

        return {
            "faithfulness": round(p_clean, 3),
            "hallucination": round(p_halluc, 3),
            "verdict": "HALLUCINATED" if p_halluc > p_clean else "FAITHFUL",
        }


if __name__ == "__main__":
    scorer = FaithfulnessScorer()
    ctx = "Refunds are processed within 30 days of purchase."
    print("Faithful:    ", scorer.score(ctx, "Refunds are given within 30 days."))
    print("Contradiction:", scorer.score(ctx, "Refunds are available for up to 90 days."))
    print("Unrelated:   ", scorer.score(ctx, "The company offers free shipping."))