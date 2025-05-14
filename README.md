# KGtelescope

KGtelescope: Mitigating Hallucination in Large Language Models via Knowledge Graph Patterns
This repository contains the code, datasets, evaluation scripts, and experimental results for the KGtelescope paper (ISWC 2025 submission).

 ## Overview
KGtelescope is a novel evaluation framework that systematically probes the factual consistency and hallucination behavior of Large Language Models (LLMs) using schema-level patterns extracted from knowledge graphs (KGs). By correlating model performance with structural features like pattern frequency and relational cardinality, KGtelescope offers interpretable insights into where and why LLMs succeed or fail in factual recall.

## Features
Schema Pattern Extraction: Abstract patterns in the form (Type, Predicate, Type) are mined from DBpedia and profiled for frequency and entity cardinality.

Factual QA Generation: Natural language questions are automatically generated from RDF triples matching extracted patterns.

Multi-Level Evaluation: LLMs are evaluated across prompting settings (zero/few-shot), similarity metrics, and cardinality/popularity strata.

Elasticity Analysis: Statistical regressions quantify how model accuracy varies with pattern frequency, identifying areas prone to hallucination.

## Repository Structure
- book-writer: It contains the python codes and results obtained from models per object-wise and subject wise cardinality
- televisionshow-person: It contains the python codes and results obtained from models per object-wise and subject wise cardinality
- supplementary files: Additional files that facilitates the reading of results obtained

The release "dbpedia2015-10 dataset files" contains the dbpedia 2015-10 snapshot in format .nt

## Datasets
DBpedia 2015 snapshot (46M triples) ensures compatibility with LLM pre-training timelines.

82,338 QA pairs covering subject-wise and object-wise queries, with cardinality bins (single, few, many) and entity popularity (PageRank quartiles) for patterns Book-Writer and TelevisionShow-Person.

## Supported Models
- LLaMA 3.2 8B
- Mistral 7B
- Gemma 4B & 12B

Inference was run locally via llama-cpp, using deterministic settings (temperature=0, no sampling) for comparability.

## Key Results
Accuracy increases significantly for frequent patterns under few-shot prompting (e.g., LLaMA 8B: +14% for (Book, author, Writer)).

Subject-wise queries (book → author) consistently outperform object-wise (author → book).

High relational complexity (e.g., many-to-many) reduces model accuracy drastically, even under high-frequency exposure.

Elasticity coefficients quantify recall sensitivity to pattern frequency: larger models (e.g., LLaMA 8B) exhibit stronger frequency leverage.
