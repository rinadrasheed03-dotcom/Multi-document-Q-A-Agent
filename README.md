# Multi-Datatype Document Question & Answer Generator

An AI-powered **Document Question & Answer Generation System** that processes multiple document formats and automatically generates document-grounded questions and answers using **NVIDIA NIM Large Language Models (LLMs)**.

The system combines document processing, text chunking, question generation, answer validation, retrieval components, reranking, and a Streamlit web interface to generate reliable question-answer pairs from uploaded documents.

---

## Features

* Supports multiple document formats
* PDF, DOCX, TXT, CSV, XLSX, and PPTX processing
* Automatic text extraction
* Text cleaning and chunking
* Configurable number of questions
* Configurable question difficulty
* Document-grounded question generation
* Automatic answer validation
* Prevents unsupported answers from being accepted
* NVIDIA NIM LLM integration
* FAISS vector retrieval
* BM25 keyword retrieval
* Reciprocal Rank Fusion (RRF)
* Cross-encoder reranking
* Duplicate-aware retrieval
* LangGraph workflow components
* Source and page metadata preservation
* Simple Streamlit web interface

---

# Supported Documents

The system supports the following document formats:

| Format | Processing Library     |
| ------ | ---------------------- |
| PDF    | PyPDF                  |
| DOCX   | python-docx            |
| TXT    | Python text processing |
| CSV    | pandas                 |
| XLSX   | pandas / openpyxl      |
| PPTX   | python-pptx            |

Multiple documents can be uploaded and processed in the same session.

---

# System Architecture

```text
                         USER
                           |
                           v
                 +---------------------+
                 |   Streamlit Web UI  |
                 |       app.py        |
                 +----------+----------+
                            |
                            | Upload Documents
                            v
              +---------------------------+
              |    Document Processing    |
              +-------------+-------------+
                            |
        +-------------------+-------------------+
        |          |          |        |        |
        v          v          v        v        v
      PDF        DOCX       TXT      CSV      XLSX/PPTX
        |          |          |        |        |
        +----------+----------+--------+--------+
                            |
                            v
                  Text Extraction
                            |
                            v
                   Text Cleaning
                            |
                            v
                    Text Chunking
                            |
                            v
                Language Detection
                            |
                            v
             +--------------------------+
             | Question Generation      |
             |        Agent              |
             +------------+-------------+
                          |
                          v
                NVIDIA NIM API
                          |
                          v
                Generated Q&A Pairs
                          |
                          v
             +--------------------------+
             | Answer Validation Agent  |
             +------------+-------------+
                          |
                 +--------+--------+
                 |                 |
              Supported         Unsupported
                 |                 |
                 v                 v
              ACCEPT             REJECT
                 |
                 v
          Requested Question Count
                 |
                 v
        +------------------------+
        | Streamlit Q&A Display  |
        +------------------------+
```

---

# RAG Retrieval Architecture

The project also contains a retrieval pipeline for document-grounded information retrieval.

```text
                    Document Chunks
                          |
             +------------+------------+
             |                         |
             v                         v
    Vector Retrieval Agent       BM25 Retrieval Agent
             |                         |
             |                         |
             +------------+------------+
                          |
                          v
                 RRF Fusion Agent
                          |
                          v
                  Reranking Agent
                          |
                          v
              Duplicate-Aware Agent
                          |
                          v
                   Answer Agent
                          |
                          v
                 Grounded Answer
```

The retrieval architecture combines semantic/vector retrieval and keyword retrieval before applying reranking and duplicate filtering.

---

# Question Generation Workflow

The main application workflow is:

```text
Upload Documents
       |
       v
Process Documents
       |
       v
Extract Text
       |
       v
Clean Text
       |
       v
Split Text into Chunks
       |
       v
Send Document Context to LLM
       |
       v
Generate Questions and Answers
       |
       v
Validate Answers
       |
       v
Reject Unsupported Questions
       |
       v
Continue Until Requested Count
       |
       v
Display Final Questions and Answers
```

---

# How the System Works

## 1. Document Upload

The user uploads one or more documents through the Streamlit interface.

Supported formats:

```text
PDF
DOCX
TXT
CSV
XLSX
PPTX
```

The application identifies the document type and uses the appropriate processing method.

---

## 2. Document Processing

Each uploaded document is processed and converted into a common text representation.

Examples:

```text
PDF   → Text extraction
DOCX  → Paragraph extraction
TXT   → Text reading
CSV   → Tabular data extraction
XLSX  → Spreadsheet extraction
PPTX  → Slide text extraction
```

Source metadata such as document name and page information is preserved where available.

---

## 3. Text Cleaning

Extracted text is cleaned before question generation.

The processing includes:

* Removing unnecessary whitespace
* Normalizing line breaks
* Normalizing tabs
* Removing excessive empty lines
* Preserving meaningful document content

---

## 4. Text Chunking

Large documents are divided into smaller chunks.

Default configuration:

```text
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
```

Chunking allows the system to process large documents efficiently while maintaining contextual continuity between neighboring chunks.

---

# Question Generation Agent

The `QuestionGenerationAgent` generates questions and answers from the extracted document context.

The agent uses the NVIDIA NIM API through the OpenAI-compatible Python client.

The LLM is instructed to:

```text
Use only the supplied document content.

Do not use outside knowledge.

Generate questions based on the document.

Generate answers supported by the document.

Return structured question-answer data.
```

The current application generates **brief-answer questions**.

The user can configure:

* Number of questions
* Difficulty level

---

# Exact Number of Questions

The application uses a target-based generation mechanism.

For example, if the user selects:

```text
Number of Questions = 10
```

the system attempts to continue generating and validating questions until:

```text
10 valid questions
```

are obtained.

The process is:

```text
Requested Count
      |
      v
Generate Questions
      |
      v
Validate Questions
      |
      +---- Valid ----> Add to Result
      |
      +---- Invalid --> Reject
      |
      v
Check Count
      |
      +---- Target reached ----> Finish
      |
      +---- Target not reached -> Generate More
```

A maximum number of generation rounds is used to prevent an infinite generation loop.

---

# Answer Validation Agent

Every generated question-answer pair is checked by the `AnswerValidationAgent`.

The validator receives:

```text
Question
Answer
Evidence
```

and determines whether the answer is supported by the supplied document content.

```text
Generated Question
        |
        v
Generated Answer
        |
        v
Evidence from Document
        |
        v
Answer Validation Agent
        |
   +----+----+
   |         |
Supported  Unsupported
   |         |
   v         v
 ACCEPT     REJECT
```

This provides an additional grounding layer and helps reduce unsupported answers.

---

# LLM and API

The system uses **NVIDIA NIM** as the LLM provider.

The application communicates with NVIDIA NIM using the OpenAI-compatible API.

### Model

```text
nvidia/nemotron-3-super-120b-a12b
```

### API Base URL

```text
https://integrate.api.nvidia.com/v1
```

### API Key

The API key is stored securely in the `.env` file:

```env
NVIDIA_API_KEY=your_nvidia_api_key
```

The API key is used for:

* Question generation
* Answer generation
* Answer validation

The LLM is accessed remotely through the NVIDIA API and is **not downloaded to the local system**.

---

# Retrieval Components

The project contains several retrieval components that can be used for document-grounded retrieval.

## Vector Retrieval

The `VectorRetrievalAgent` uses **FAISS** for vector similarity search.

```text
Query
  |
  v
Query Vector
  |
  v
FAISS Index
  |
  v
Top-K Relevant Chunks
```

---

## BM25 Retrieval

The `BM25RetrievalAgent` provides keyword-based retrieval.

```text
Query
  |
  v
BM25 Search
  |
  v
Keyword-Relevant Chunks
```

BM25 complements vector retrieval by finding content based on important keywords and terms.

---

## Reciprocal Rank Fusion

The `RRFFusionAgent` combines the results from:

```text
Vector Retrieval
       +
BM25 Retrieval
       |
       v
Reciprocal Rank Fusion
       |
       v
Combined Ranking
```

This allows the system to combine semantic and keyword-based retrieval results.

---

## Cross-Encoder Reranking

The `RerankingAgent` uses a CrossEncoder model to rerank retrieved chunks.

Model:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The model evaluates:

```text
Query + Document Chunk
```

and produces a relevance score.

The highest-scoring chunks are selected.

---

## Duplicate-Aware Retrieval

The `DuplicateAwareAgent` helps identify redundant or highly similar content.

The system uses duplicate detection to avoid repeatedly selecting essentially the same information.

Configuration:

```text
DUPLICATE_THRESHOLD=0.94
```

---

# Embedding Service

The project contains an `EmbeddingService` used by the vector retrieval component.

The current implementation generates deterministic vector representations using hashing rather than a semantic embedding model.

Therefore, the following configuration:

```env
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

is currently available as a configurable model setting, but the present embedding implementation does not directly use this model to create the vectors.

The architecture is designed so that a semantic embedding model can be integrated in the future.

---

# Reranking Model

The current reranking model is:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

It is loaded through:

```text
Sentence Transformers
```

The model is downloaded automatically from Hugging Face when it is first required and is not already available in the local cache.

---

# Technologies Used

| Technology            | Purpose                                            |
| --------------------- | -------------------------------------------------- |
| Python                | Main programming language                          |
| Streamlit             | Web interface                                      |
| NVIDIA NIM            | Large Language Model API                           |
| OpenAI SDK            | API client for NVIDIA's OpenAI-compatible endpoint |
| LangGraph             | Agent/workflow orchestration                       |
| FAISS                 | Vector similarity retrieval                        |
| BM25                  | Keyword retrieval                                  |
| RRF                   | Retrieval result fusion                            |
| Sentence Transformers | Cross-encoder reranking                            |
| PyTorch               | ML runtime                                         |
| PyPDF                 | PDF processing                                     |
| python-docx           | DOCX processing                                    |
| python-pptx           | PPTX processing                                    |
| pandas                | CSV/XLSX processing                                |
| openpyxl              | Excel processing                                   |
| python-dotenv         | Environment configuration                          |
| Lingua                | Language detection                                 |
| NumPy                 | Numerical operations                               |

---

# Configuration

Create a `.env` file in the project root.

Example:

```env
# NVIDIA NIM
NVIDIA_API_KEY=your_nvidia_api_key
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=nvidia/nemotron-3-super-120b-a12b

# LLM
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=4096
LLM_MAX_RETRIES=3

# Embedding
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Reranker
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Chunking
CHUNK_SIZE=1000
CHUNK_OVERLAP=150

# Retrieval
VECTOR_TOP_K=20
BM25_TOP_K=20
RRF_TOP_K=20
RERANK_TOP_K=10
FINAL_CONTEXT_K=5

# RRF
RRF_CONSTANT=60

# Duplicate Detection
DUPLICATE_THRESHOLD=0.94
```

---

# Installation

## Requirements

The system requires:

```text
Python 3.12 or later
```

An internet connection is required for:

* Installing Python packages
* Accessing NVIDIA NIM
* Downloading the reranking model when required

---

## Create Virtual Environment

### Windows

```bash
python -m venv .venv
```

Activate:

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
```

Activate:

```bash
source .venv/bin/activate
```

---

# Install Dependencies

Install the project dependencies using:

```bash
pip install -e .
```

Or install the required packages manually:

```bash
pip install streamlit openai langgraph faiss-cpu rank-bm25 sentence-transformers torch numpy pypdf python-docx python-pptx pandas openpyxl python-dotenv lingua-language-detector
```

---

# Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your web browser.

---

# Project Structure

```text
Multi-datatype_questin_answer RAG Agent/
│
├── app.py
├── README.md
├── pyproject.toml
├── uv.lock
├── .env
├── .gitignore
│
├── src/
│   │
│   ├── agents/
│   │   ├── answer_agent.py
│   │   ├── answer_validation_agent.py
│   │   ├── bm25_retrieval_agent.py
│   │   ├── duplicate_aware_agent.py
│   │   ├── pdf_processing_agent.py
│   │   ├── question_generation_agent.py
│   │   ├── reranking_agent.py
│   │   ├── rrf_fusion_agent.py
│   │   └── vector_retrieval_agent.py
│   │
│   ├── graph/
│   │   ├── nodes.py
│   │   ├── state.py
│   │   └── workflow.py
│   │
│   ├── ingestion/
│   │   ├── csv_loader.py
│   │   ├── docx_loader.py
│   │   ├── document_loader.py
│   │   ├── embeddings.py
│   │   ├── pdf_loader.py
│   │   ├── pptx_loader.py
│   │   ├── txt_loader.py
│   │   ├── xlsx_loader.py
│   │   └── chunker.py
│   │
│   ├── models/
│   │   └── qa_schema.py
│   │
│   ├── retrieval/
│   │   ├── bm25_index.py
│   │   ├── metadata_filter.py
│   │   └── vector_store.py
│   │
│   ├── utils/
│   │   ├── citation_utils.py
│   │   ├── language_detector.py
│   │   └── text_utils.py
│   │
│   └── config.py
│
└── tests/
    ├── test_duplicate_detection.py
    ├── test_reranker.py
    ├── test_retrieval.py
    └── test_rrf.py
```

---

# Complete System Flow

```text
                         USER
                           |
                           v
                  Streamlit Interface
                           |
                           v
                  Upload Documents
                           |
                           v
                 Document Processing
                           |
                           v
                  Text Extraction
                           |
                           v
                    Text Cleaning
                           |
                           v
                     Chunking
                           |
                           v
                  Language Detection
                           |
                           v
              Question Generation Agent
                           |
                           v
                    NVIDIA NIM
                           |
                           v
                  Generated Q&A
                           |
                           v
              Answer Validation Agent
                           |
                 +---------+---------+
                 |                   |
              Valid                Invalid
                 |                   |
                 v                   X
             Accepted             Rejected
                 |
                 v
           Target Count Reached?
                 |
            +----+----+
            |         |
           No        Yes
            |         |
            v         v
       Generate More  Final Results
                        |
                        v
                 Streamlit Display
                        |
                        v
                  Questions + Answers
```

---

# Security

Never commit the `.env` file containing your NVIDIA API key.

Add the following to `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
*.pyc
.pytest_cache/
```

The API key should only be stored locally in the `.env` file or in a secure deployment environment.

---

# Testing

The project contains tests for retrieval and ranking components.

Run:

```bash
pytest
```

Tested components include:

```text
Retrieval
Reranking
RRF Fusion
Duplicate Detection
```

---

# Limitations

* The current question-generation flow processes document chunks directly rather than performing retrieval before every question-generation request.
* The current embedding implementation uses deterministic hash-based vectors.
* The configured `EMBEDDING_MODEL` is therefore not currently used to generate the FAISS vectors.
* NVIDIA NIM requires a valid API key and internet connectivity.
* The CrossEncoder reranking model requires an initial download if it is not already cached.
* The current application primarily generates brief-answer questions.

---

# Future Enhancements

Future versions can include:

* True semantic embedding models
* Full RAG integration with question generation
* Multiple question types
* Multiple-choice question generation
* True/False questions
* Fill-in-the-blank questions
* Improved multilingual support
* Persistent vector databases
* Question difficulty classification
* Improved duplicate question detection
* PDF/Word/Excel/JSON export
* Citation verification
* Document indexing and caching
* Authentication
* Cloud deployment

---

# License

Add your preferred license information here.

---

# Author

Rinad R

Multi-Datatype Document Question & Answer Generator
