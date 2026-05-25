# RAG Reliability Inspector

A reliability and faithfulness inspection layer for Retrieval-Augmented Generation (RAG) systems. It detects when an LLM's answer is not grounded in its retrieved documents — and diagnoses *why* the failure happened: retrieval, generation, or both.

Think of it as **observability for RAG answer quality** — the way Sentry surfaces code errors, this surfaces unreliable AI answers before they reach a user.

## The Problem

LLMs hallucinate even within RAG pipelines. Hallucinations persist for three reasons:

1. **Retrieval failure** — the wrong documents were fetched.
2. **Generation infidelity** — the right documents were fetched, but the model ignored or contradicted them.
3. **Partial grounding** — most of the answer is supported, but an unsupported claim is slipped in.

For companies deploying customer-facing RAG chatbots, a single wrong answer can mean a lost customer or legal liability. Most teams ship these answers blind, with no visibility into reliability.

## Our Approach

A two-stage inspection layer, using a different model for each stage because they are fundamentally different problems:

| Stage | Question | Model |
|-------|----------|-------|
| 1. Retrieval Relevance | Is the retrieved context relevant to the query? | MS-MARCO cross-encoder |
| 2. Generation Faithfulness | Is the answer supported by the context? | **Fine-tuned RoBERTa** (trained on RAGTruth) |

The faithfulness model is fine-tuned to classify whether an answer sentence is grounded (clean) or hallucinated, catching contradictions, baseless information, and unsupported content.

## Results

Faithfulness detection on the RAGTruth QA benchmark:

| Model | F1 | Precision | Recall |
|-------|----|-----------|--------|
| Off-the-shelf NLI (zero-shot baseline) | 0.43 | — | — |
| Fine-tuned RoBERTa (RAGTruth) | 0.577 | 0.505 | 0.674 |
| Fine-tuned RoBERTa (+ contradiction examples) | **0.590** | 0.506 | 0.708 |

Fine-tuning improved F1 by ~37% over the zero-shot baseline.

## Engineering Notes

- Initially attempted fine-tuning DeBERTa-v3 but hit training instability (NaN loss from its disentangled-attention implementation on our stack); switched to RoBERTa, which trained stably.
- Handled the class imbalance (~11% hallucinated) by oversampling the minority class.
- Augmented the RAGTruth training data with synthetic contradiction examples to improve detection of direct factual/numeric contradictions.

## Project Structure

- `src/retrieval_scorer.py` — retrieval relevance scoring
- `src/faithfulness_scorer.py` — faithfulness scoring (fine-tuned model)
- `src/inspector.py` — combines both stages into a diagnostic verdict
- `phase*.py` — data exploration, evaluation, and training-data scripts
- `evaluation_results.csv`, `best_threshold.json` — benchmark outputs

## Note on Large Files

The fine-tuned model (~400 MB), the RAGTruth dataset, and the virtual environment are excluded from this repo (see `.gitignore`). The model is shared separately due to size; the dataset can be downloaded from [ParticleMedia/RAGTruth](https://github.com/ParticleMedia/RAGTruth).

## Tech Stack

`transformers`, `sentence-transformers`, `torch`, `scikit-learn`, RAGTruth, Streamlit (dashboard).

## Status

- [x] Two-stage inspector with retrieval + faithfulness scoring
- [x] Fine-tuned faithfulness model (F1 0.590 on RAGTruth)
- [x] End-to-end diagnostic pipeline
- [ ] Streamlit dashboard (playground, monitoring feed, performance tab)
- [ ] SDK adoption snippet

## Team

Bilal, Raahim, Unnus