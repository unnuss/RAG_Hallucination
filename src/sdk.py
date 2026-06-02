"""
RAG Reliability SDK.

A lightweight wrapper a developer adds to their existing RAG pipeline.
Once wrapped, every answer their system produces is automatically
inspected for reliability and logged.

This is the integration surface: in production, the logged results would
be sent to a server/dashboard. Here we log locally to a JSONL file that
the dashboard reads, demonstrating the full loop on one machine.
"""

import json
import time
from datetime import datetime
from typing import Callable, List, Union

from src.inspector import Inspector


class ReliabilityMonitor:
    """
    Wraps a RAG pipeline so every answer is automatically inspected.

    Usage (the developer's integration, just a few lines):

        monitor = ReliabilityMonitor()

        @monitor.track
        def answer_question(query):
            context = my_retriever(query)
            answer = my_llm(query, context)
            return query, context, answer
    """

    def __init__(self, log_path: str = "monitoring_log.jsonl"):
        self.inspector = Inspector()
        self.log_path = log_path

    def inspect(self, query: str,
                context: Union[str, List[str]],
                answer: str) -> dict:
        """Inspect a single RAG output and log the result."""
        result = self.inspector.evaluate(query, context, answer)

        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "query": query,
            "answer": answer,
            "retrieval_relevance": result.retrieval_relevance,
            "faithfulness_score": result.faithfulness_score,
            "hallucination_score": result.hallucination_score,
            "status": result.overall_status,
            "diagnosis": result.diagnosis,
        }

        # Append to the local log (the dashboard reads this).
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        return record

    def track(self, rag_fn: Callable):
        """
        Decorator. Wrap a function that returns (query, context, answer),
        and every call is automatically inspected + logged.
        """
        def wrapper(*args, **kwargs):
            query, context, answer = rag_fn(*args, **kwargs)
            record = self.inspect(query, context, answer)
            return answer, record
        return wrapper