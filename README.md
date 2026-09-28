# Local RAG: Interactive QA and Summarization for Your PDFs

> **Note:** Originally developed as **`course-rag`** to help students query lecture slides and course materials, this project has evolved into **`local-rag`** (or **`pdf-rag`**)—a fully adaptable, offline RAG system for analyzing any collection of local PDF documents.

A local, command-line Retrieval-Augmented Generation (RAG) assistant for querying and summarizing your PDFs. Ask conceptual questions, generate document summaries, or hold multi-turn conversations grounded directly in your files. Every answer is explicitly prompted to cite source filenames and page numbers.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Interactive Chat](#interactive-chat)
  - [Single-Shot CLI Commands](#single-shot-cli-commands)
  - [Chat Commands Reference](#chat-commands-reference)
- [Managing Topics and Folders](#managing-topics-and-folders)
- [Configuration and Model Switching](#configuration-and-model-switching)
- [Database Synchronization](#database-synchronization)
- [Architecture Overview](#architecture-overview)
- [Project Layout and Testing](#project-layout-and-testing)

---

## Prerequisites

Before running the application, ensure you have installed:

1. **[uv](https://astral.sh/uv)** (Fast Python package manager):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. **[Ollama](https://ollama.com/)** (Local LLM runner):
   Download and install Ollama for your operating system.

---

## Quick Start

Get up and running in five steps:

1. **Pull the default LLM model:**
   ```bash
   ollama pull openbmb/minicpm5-2b:latest
   ```

2. **Install project dependencies:**
   ```bash
   uv sync
   ```
   > **macOS Note:** If `uv sync` completes but running commands produces `ModuleNotFoundError: local_rag`, unhide the package path file:
   > ```bash
   > chflags nohidden .venv/lib/python*/site-packages/local_rag.pth
   > ```

3. **Start the Ollama service:**
   Make sure the desktop application is running, or execute:
   ```bash
   ollama serve
   ```

4. **Launch the assistant:**
   ```bash
   uv run python query_engine.py
   ```
   *(On first startup, the app automatically downloads the `all-MiniLM-L6-v2` embedding model and indexes files in the `data/` directory.)*

---

## Usage

### Interactive Chat

Launch the interactive prompt and enter chat commands or plain-text questions:

```text
/files
/pdf document_a.pdf
Summarize this PDF.
What are the primary concepts discussed in this file?
Explain that in simpler terms.
/pdf document_b.pdf
What is the core argument in this document?
/all
/summary Summarize both documents and compare their main findings.
```

### Single-Shot CLI Commands

Execute quick queries or generate summaries directly from your terminal:

```bash
# Query a specific document
uv run pdf-rag ask "What are the key takeaways?" --pdf "document_a.pdf"

# Generate a combined summary across all active PDFs
uv run pdf-rag ask "Compare the methodology across all files." --summary
```

### Chat Commands Reference

| Command | Description |
| :--- | :--- |
| `/topics` | List available document topics (folders). |
| `/use <TOPIC>` | Index and switch to a topic; restores recent chat history. |
| `/files` | List PDFs in the active topic. |
| `/pdf <FILENAME>` | Target questions and summaries toward a specific PDF. |
| `/all` | Target all PDFs in the active topic. |
| `/summary [request]` | Perform full-text extraction and generate document summaries. |
| `/index` | Refresh the active topic index after file changes and reset history. |
| `/reset` | Clear conversation history for the current topic. |
| `/help` | Display available commands. |
| `/quit` | Exit the interactive session. |

---

## Managing Topics and Folders

Organize your PDFs by placing them into subdirectories inside `data/`. Each folder acts as an isolated topic:

```text
data/
├── document_a.pdf          # Topic: . (default root)
├── document_b.pdf
├── research/               # Topic: research
│   └── paper_01.pdf
├── finance/                # Topic: finance
│   └── Q3_report.pdf
└── project_x/sub_folder/   # Topic: project_x/sub_folder
    └── specs.pdf
```

### Topic CLI Commands

```bash
# List all discovered topics
uv run pdf-rag topics

# Index all topic directories at once
uv run pdf-rag index --all

# Launch chat within a specific topic
uv run pdf-rag chat --topic research
```

### Switching Topics in Chat

```text
/topics
/use finance
What were the quarterly revenue numbers?
/use research
Explain the algorithm described in paper_01.pdf.
/use .
```

*Each topic maintains its own vector index, PDF selection, and recent conversation history (retained in memory for up to three exchanges).*

---

## Configuration and Model Switching

### Changing the LLM Model

You can use any local model installed via `ollama pull`. Specify models using CLI flags or environment variables:

```bash
# Override model for an interactive session:
uv run pdf-rag chat --model gemma4:e4b

# Override model for a single query:
uv run pdf-rag ask "Explain this term." --model gemma4:e4b

# Set model globally using an environment variable:
OLLAMA_MODEL=gemma4:e4b uv run python query_engine.py
```

### Custom Directory Paths

```bash
uv run pdf-rag chat --data /path/to/custom_pdfs --db /path/to/custom_db --topic research
```

- `OLLAMA_MODEL`: Overrides default answer model string (`openbmb/minicpm5-2b:latest`).
- `OLLAMA_URL`: Overrides target Ollama server address.

---

## Database Synchronization

The persistent vector database resides in `chroma_db/`. Re-indexing happens automatically when starting a session, switching topics, or executing explicit indexing commands.

### File Modification Rules

| Event in `data/` | Re-indexing Behavior |
| :--- | :--- |
| **Add new PDF** | Extracts text, generates embeddings, adds to index. |
| **Edit existing PDF** | Detects change via SHA-256 hash; replaces old embeddings. |
| **Unchanged PDF** | Skipped automatically. |
| **Delete PDF** | Removes associated chunks from topic index. |
| **Rename PDF** | Treated as a deletion followed by addition; re-embeds text. |
| **Move PDF between folders** | Removed from origin topic; indexed under target topic. |

> **Performing a Clean Rebuild:** To completely rebuild your vector database, stop the application, delete or rename `chroma_db/`, and run `uv run python indexer.py --all`.

---

## Architecture Overview

1. **Extraction:** Page-by-page text parsing via `pypdf` and `fontTools`.
2. **Embedding & Storage:** Overlapping text chunks are embedded using Sentence Transformers (`all-MiniLM-L6-v2`) and stored in isolated Chroma collections per topic.
3. **Retrieval:** Relevant chunks (up to 6 per query) are fetched based on semantic similarity.
4. **Summarization Engine:** Reads and processes indexed chunks in bounded batches, generating document-level summaries before forming final answers.
5. **Generation & Grounding:** Local LLM constructs answers using retrieved evidence, formatted with mandatory source citations.

---

## Project Layout and Testing

```text
src/local_rag/library.py     # PDF parsing, indexing, and retrieval pipeline
src/local_rag/assistant.py   # RAG prompt engine, history, and generation logic
src/local_rag/cli.py         # Terminal commands and interactive prompt UI
indexer.py                   # CLI wrapper for indexing workflows
query_engine.py              # CLI wrapper for query & chat workflows
tests/                       # Automated offline test suite
chroma_db/                   # Local persistent vector store (gitignored)
.tmp/                        # Local cache directory (gitignored)
```

Run unit and integration tests (requires no active Ollama server):

```bash
uv run python -m unittest discover -s tests -v
```