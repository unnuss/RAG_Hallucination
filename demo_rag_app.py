"""
Demo RAG app — simulates a company's customer-support chatbot.

It has a small knowledge base, retrieves relevant docs, and generates
answers (some intentionally hallucinated to show detection).

Wrapped with our SDK, every answer is automatically inspected + logged.
Run with:  python3 -m demo_rag_app
"""

from src.sdk import ReliabilityMonitor

# ---- The company's knowledge base (their documents) ----
KNOWLEDGE_BASE = {
    "refund": "Refunds are processed within 30 days of the purchase date.",
    "warranty": "The warranty covers the product for 12 months from purchase.",
    "shipping": "Standard shipping takes 5 to 7 business days.",
    "hours": "Our support office is open Monday to Friday, 9am to 5pm.",
    "membership": "Annual membership costs 50 dollars per year.",
}

# ---- A naive retriever: pick the doc whose keyword is in the query ----
def retrieve(query: str) -> str:
    q = query.lower()
    for keyword, doc in KNOWLEDGE_BASE.items():
        if keyword in q:
            return doc
    # Fallback: wrong doc (simulates a retrieval miss)
    return KNOWLEDGE_BASE["shipping"]

# ---- A "generator" that sometimes hallucinates (scripted for the demo) ----
# In a real app this is an LLM call. We script answers to show both
# faithful and hallucinated outputs.
SCRIPTED_ANSWERS = {
    "What is your refund policy?":
        "Customers can request a refund within 30 days of purchase.",          # faithful
    "How long is the warranty?":
        "The product is covered under warranty for 24 months.",                # hallucination (24 vs 12)
    "How long does shipping take?":
        "Standard shipping takes 5 to 7 business days.",                       # faithful
    "What are your office hours?":
        "Our office is open 24/7 including weekends and holidays.",            # hallucination
    "How much is membership?":
        "Membership costs 50 dollars per year and includes free shipping.",    # partial: adds unsupported claim
}

# ---- Set up the monitor and wrap our RAG pipeline ----
monitor = ReliabilityMonitor(log_path="monitoring_log.jsonl")

@monitor.track
def answer_question(query: str):
    context = retrieve(query)
    answer = SCRIPTED_ANSWERS.get(query, "I'm not sure about that.")
    return query, context, answer


if __name__ == "__main__":
    print("Running demo RAG app through the Reliability Monitor...\n")

    for query in SCRIPTED_ANSWERS:
        answer, record = answer_question(query)
        print(f"Q: {query}")
        print(f"A: {answer}")
        print(f"   -> Status: {record['status']}  "
              f"(faithfulness {record['faithfulness_score']:.2f})")
        print(f"   -> {record['diagnosis']}\n")

    print("All answers inspected and logged to monitoring_log.jsonl")
    print("Open the dashboard's Monitoring Feed to see them.")