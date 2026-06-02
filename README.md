# RAG Reliability Inspector

**Observability for RAG answer quality. Detect and diagnose hallucinations before they reach your users.**

A monitoring tool for production RAG (Retrieval-Augmented Generation) systems that automatically catches hallucinations in AI chatbot answers. Think Sentry, but for AI answer quality.

## What this is

LLMs powering customer-facing RAG chatbots still hallucinate even when given the correct source documents — they override retrieved context with training priors, embellish, and misread negations. Companies running these chatbots have no systematic way to know when their AI is lying.

This project is a working prototype of a monitoring layer for RAG systems:

- A **fine-tuned RoBERTa** model that scores answer faithfulness to retrieved context, benchmarked on RAGTruth
- A **two-stage diagnostic inspector** that separates retrieval failures from generation failures
- A **lightweight SDK** that wraps any existing RAG pipeline with a single decorator
- A **dashboard** for engineering teams to monitor AI answer reliability in real time

## Results

Evaluation on the RAGTruth QA test set (300 examples):

| Metric | Zero-shot baseline | Fine-tuned RoBERTa |
|---|---|---|
| F1 | 0.43 | **0.63** |
| Precision | 0.29 | **0.58** |
| Recall | 0.79 | **0.70** |
| False Positives | 108 | **29** |

**~47% relative F1 improvement**, with false alarms reduced by 73%.

Confusion matrix at best threshold:
- True Positives: 40
- False Positives: 29
- False Negatives: 17
- True Negatives: 214
- Accuracy: 85%

## How it works

Two stages, each one a separate model:

**Stage 1 — Retrieval Relevance.** A pre-trained cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) scores how relevant the retrieved context is to the user's query. Low score → retrieval failure.

**Stage 2 — Faithfulness.** A fine-tuned RoBERTa scores whether the AI's answer is grounded in the retrieved context. Low score → generation hallucination.

Combined, the system produces one of four diagnoses: **Reliable**, **Low (retrieval failure)**, **Low (generation hallucination)**, or **Very Low (both)**.

## SDK integration

Developers wrap their existing RAG function with one decorator:

```python
