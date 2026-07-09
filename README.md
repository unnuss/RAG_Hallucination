RAG Reliability Inspector

An observability layer for Retrieval-Augmented Generation (RAG) systems. Automatically detect, diagnose, and monitor hallucinations before they reach users.

Think Sentry, but for AI answer quality.

RAG Reliability Inspector is a production-inspired monitoring tool for RAG applications. Instead of generating answers, it evaluates whether those answers are actually supported by the retrieved context, helping engineering teams identify and diagnose hallucinations before they reach end users.

⸻

Why this matters

Retrieval-Augmented Generation (RAG) is widely used to reduce hallucinations by retrieving relevant documents before an LLM generates a response.

In practice, retrieval alone isn’t enough.

Even when the correct document is retrieved, LLMs may ignore it, rely on information from their pretraining, introduce unsupported details, or contradict the retrieved context altogether. Existing evaluation tools often provide a single reliability score but don’t explain why an answer failed.

RAG Reliability Inspector addresses this by separating retrieval failures from generation failures, allowing engineers to identify the source of unreliable responses and debug their systems more effectively.

⸻

Features

* Fine-tuned RoBERTa classifier for answer faithfulness, benchmarked on RAGTruth
* Two-stage diagnostic pipeline separating retrieval failures from generation hallucinations
* Lightweight Python SDK that integrates into existing RAG applications with a single decorator
* Real-time Streamlit dashboard for monitoring AI answer quality
* Automatic logging and reliability scoring for every generated response
* Local inference with deterministic evaluation (no LLM-as-a-judge required)

⸻

Architecture

                 User Query
                      │
                      ▼
                 Retriever
                      │
              Retrieved Context
                      │
                      ▼
                    LLM
                      │
              Generated Answer
                      │
                      ▼
        RAG Reliability Inspector
        ┌────────────────────────┐
        │ Retrieval Relevance    │
        │ Faithfulness           │
        └────────────────────────┘
                      │
                      ▼
       Reliability Diagnosis
                      │
                      ▼
        Dashboard + Monitoring Logs

⸻

Results

Evaluation on the RAGTruth QA benchmark (300-example test set):

Metric	Zero-shot Baseline	Fine-tuned RoBERTa
F1 Score	0.43	0.63
Precision	0.29	0.58
Recall	0.79	0.70
False Positives	108	29

Key improvements

* 47% relative improvement in F1 score
* 73% reduction in false positives
* 85% overall accuracy

Confusion Matrix

	Predicted Hallucination	Predicted Faithful
Actual Hallucination	40	17
Actual Faithful	29	214

⸻

How it works

RAG Reliability Inspector evaluates every response using two independent stages.

Stage 1: Retrieval Relevance

A pre-trained cross-encoder (cross-encoder/ms-marco-MiniLM-L-6-v2) evaluates whether the retrieved document is relevant to the user’s query.

A low relevance score indicates a retrieval failure, where the system fetched the wrong context before generation.

⸻

Stage 2: Faithfulness

A fine-tuned RoBERTa classifier evaluates whether the generated answer is supported by the retrieved context.

The model was trained on more than 34,000 context-sentence pairs, including the RAGTruth benchmark and additional synthetic contradiction examples.

Low faithfulness indicates a generation hallucination, where the model ignored or contradicted the retrieved evidence.

⸻

Diagnostic Output

Rather than returning a single reliability score, the inspector identifies which component failed.

Retrieval	Faithfulness	Diagnosis
✓	✓	Reliable
✗	✓	Retrieval Failure
✓	✗	Generation Hallucination
✗	✗	Both Retrieval and Generation Failed

⸻

SDK Integration

Integrating the inspector into an existing RAG application requires only a single decorator.

from rag_reliability import ReliabilityMonitor
monitor = ReliabilityMonitor()
@monitor.track
def answer_question(query):
    context = my_retriever(query)
    answer = my_llm(query, context)
    return query, context, answer

Every interaction is automatically captured, evaluated, logged, and made available through the monitoring dashboard without requiring changes to the existing application logic.

⸻

Dashboard

The accompanying Streamlit dashboard provides a real-time view of answer reliability across a RAG system.

It includes:

* Playground for inspecting individual responses
* Monitoring Feed with live reliability logs and diagnostics
* Performance Dashboard displaying benchmark metrics and confusion matrices

This enables engineering teams to move beyond simple pass/fail scoring and understand why unreliable responses occur.

⸻

Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* RoBERTa
* Sentence Transformers
* Streamlit
* scikit-learn
* Pandas
* RAGTruth Benchmark

⸻

Future Work

* Span-level attribution to identify the exact unsupported claim
* Domain-specific models for legal, healthcare, and finance
* Hosted production deployment for real-time monitoring at scale
* Expanded benchmark evaluation across additional RAG datasets

⸻

Project Motivation

This project was developed as part of a Deep Learning course to explore a practical challenge in modern LLM systems: building trustworthy AI isn’t only about generating better answers—it’s also about building the infrastructure needed to monitor, diagnose, and understand when those systems fail.
