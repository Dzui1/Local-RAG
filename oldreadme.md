# PDF RAG: questions and conversations about your PDFs

A local command-line assistant for PDFs on any subject. Ask what a concept means,
request a document summary, or continue a conversation with follow-up questions.
Answers are prompted to cite PDF filenames and physical page numbers.

## Start with the two PDFs already in `data`

Run these commands from the project directory:

```bash
uv sync
# Start Ollama if it isn't already running (or open the Ollama app):
ollama serve
```

In another terminal:

```bash
uv run python query_engine.py
```

Chat automatically indexes the selected topic. The first index downloads the
`all-MiniLM-L6-v2` embedding model. Ollama uses the already-installed
`openbmb/minicpm5-2b:latest` by default.
On a new machine install it with:

```bash
ollama pull openbmb/minicpm5-2b:latest
```

The answer model can be overridden with `--model` or `OLLAMA_MODEL`; no automatic
model switching occurs. It is separate from the embedding model, so changing the
answer model does not require rebuilding the vector database. See [Model
switching and thinking](#model-switching-and-thinking) for examples.

Try this conversation (enter filenames exactly, without surrounding quotes):

```text
/files
/pdf How to study cities_LeGates.pdf
Summarize this PDF.
What approaches does LeGates suggest for studying cities?
Explain that more simply.
/pdf What is urban about critical urban theory .pdf
What does critical urban theory mean in this text?
/all
/summary Summarize both PDFs and explain how their approaches differ.
```

Or ask a single question:

```bash
uv run pdf-rag ask "What does critical urban theory mean?" \
  --pdf "What is urban about critical urban theory .pdf"
uv run pdf-rag ask "Summarize both PDFs and compare their main ideas." --summary
```

## Organize conversations by topic

Every folder containing PDFs is a separate topic. The topic name is its path
relative to `data`. PDFs directly inside `data` belong to the default topic `.`.
Only PDFs directly in a given folder are included in that topic; nested folders
have their own topics. Filenames ending in `.pdf` or `.PDF` are supported.
Without `--topic`, the assistant opens `.` when present, otherwise the first
available topic and prints its name.

```text
data/
  How to study cities_LeGates.pdf              # topic: .
  What is urban about critical urban theory .pdf
  biology/                                   # topic: biology
    cells.pdf
  history/                                   # topic: history
    reading.pdf
  work/project-a/                            # topic: work/project-a
    report.pdf
```

```bash
uv run pdf-rag topics
uv run pdf-rag index --all
uv run pdf-rag chat --topic biology
```

Switch at any time inside chat:

```text
/topics
/use history
What are the main arguments?
/use biology
Explain the previous answer more simply.
/use .
```

Each topic has a separate index, recent conversation history, and PDF selection.
Switching back restores that topic's conversation during the current session.
History stays in memory (the last three exchanges); exiting starts a fresh
conversation next time. Indexes persist on disk. Selecting a different PDF or
using `/all` clears the current topic's history to avoid mixing document context.
Topic switching is explicit so an unrelated question cannot silently switch sources.

### Chat commands

| Command | Action |
| --- | --- |
| `/topics` | List PDF folders |
| `/use TOPIC` | Index and switch to a topic; resume its history |
| `/files` | List PDFs in the active topic |
| `/pdf FILENAME` | Focus questions and summaries on one PDF |
| `/all` | Use all PDFs in the active topic |
| `/summary [request]` | Read and summarize all text in the selected PDF(s) |
| `/index` | Refresh the active topic after changing PDFs; clear its history |
| `/reset` | Start a fresh conversation in the active topic |
| `/help` | Show commands |
| `/quit` | Exit |

## Adding, removing, and reorganizing PDFs

The vector database in `chroma_db/` is a persisted index, not a live mirror of
`data/`. There is no background file watcher. Synchronization happens when you
start chat, run a one-shot `ask`, switch with `/use TOPIC`, or explicitly index.
Chat startup and `ask` refresh only the selected topic; `/use` refreshes the
destination topic.

After changing PDFs during a conversation, run `/index` to refresh that topic
and clear its history. After reorganizing several folders, run:

```bash
uv run python indexer.py --all
```

| Change in `data/` | What the next index does |
| --- | --- |
| Add a PDF | Extracts its text, creates embeddings, and adds them to that topic. |
| Edit or replace a PDF under the same filename | Detects changed file contents using SHA-256; replaces that PDF's old chunks and embeddings. |
| Leave a PDF unchanged | Skips it when its stored index is complete. |
| Remove a PDF from a topic that still has other PDFs | Deletes the removed PDF's stored chunks and embeddings. |
| Rename a PDF | Treats this as removing the old filename and adding the new one; embeds it again. |
| Move a PDF between topics | Removes it from the old topic when that topic is refreshed and nonempty; indexes it in the destination topic. Refresh both, or use `--all`. |
| Empty, delete, or rename an entire topic folder | The old topic becomes unavailable, but its collection remains on disk. A renamed folder is indexed as a new topic. |

**Current cleanup limitation:** `--all` discovers only folders that currently
contain PDFs. It does not purge collections for empty, deleted, or renamed topic
folders. If all PDFs are removed from `data`, indexing reports that no PDFs were
found and leaves the existing database intact. These old collections cannot be
queried through the assistant while their topics are absent, but still take space.

Collections are identified by the absolute data-directory path plus the topic's
relative folder path. Moving the entire project or changing `--data` to another
location therefore creates separate collections, even for identical PDFs.
For a clean rebuild after a major reorganization, stop the assistant, move
`chroma_db/` aside as a backup, and run `uv run python indexer.py --all`.
This regenerates the index from the current PDFs; it does not modify the PDFs.
If you use `--db`, apply these steps to that database directory instead.

During an already-open chat, all-PDF queries can still retrieve removed or
outdated passages until the topic is refreshed. An external indexing command
does not clear that running chat's history; use `/index` there or restart chat.
`/use` preserves history, so use `/reset` after returning to a topic whose
documents changed.

Changing the answer model does not affect this synchronization: the vector
database stores PDF embeddings, while Ollama is called later to write an answer.
Changing the embedding model in `src/courses_rag/library.py` is different. It
changes the vectors used for search, so move `chroma_db/` aside and rebuild with
`uv run python indexer.py --all` after changing the embedding model.

## How it works

- `pypdf` and `fontTools` extract text page by page.
- Overlapping text chunks are embedded with Sentence Transformers and stored in
  Chroma, with a separate collection for each topic and data directory.
- Questions retrieve up to six relevant chunks, filtered to the selected PDF if
  applicable. Recent conversation helps retrieve evidence for follow-up questions.
- Summary requests (`summarize`, `summary`, `overview`, or `/summary`) process
  **every indexed chunk** in bounded sections, then summarize each PDF separately
  before answering the request using all document summaries. Comparisons happen
  only after every selected document has been summarized.
  Long summaries take multiple local model calls and print progress.
- Ollama generates answers using PDF evidence and citation instructions.
- File hashes detect changed PDFs. Reindexing replaces changed files, removes
  deleted files from nonempty topics, and skips unchanged complete indexes.
  Empty/deleted topics are unavailable to query even if old index data remains.

Summaries cover all extracted text, but generated answers and citations can still
be wrong; verify important claims against the cited pages. Each answer also includes a source list generated directly
from the records supplied to the model, even if the model omits inline citations.
Image-only PDFs need OCR before indexing. Figures and images are not interpreted. Physical PDF page
numbers may differ from printed page numbers. Initial setup downloads packages
and model weights; PDF extraction, embeddings, and default Ollama requests run
locally. Summaries need more time than ordinary questions.

## Model switching and thinking

The default answer model is set in
[`src/courses_rag/assistant.py`](/Users/dzui_/Documents/Software/Coding/Courses-RAG/src/courses_rag/assistant.py:45):

```python
self.model = model or os.getenv(
    "OLLAMA_MODEL", "openbmb/minicpm5-2b:latest")
```

The three ways to choose a model are, in order of precedence:

```bash
# One interactive session:
uv run python query_engine.py --model gemma4:e4b

# One question:
uv run pdf-rag ask "What does this term mean?" --model gemma4:e4b

# The process environment (no source edit):
OLLAMA_MODEL=gemma4:e4b uv run python query_engine.py
```

Replace `gemma4:e4b` with the exact name shown by `ollama list`. The model is
selected when the process starts, so restart the chat to switch models. The
`--model` option takes precedence over `OLLAMA_MODEL`, which takes precedence
over the default string in `assistant.py`. `--model` is available for `ask` and
`chat`; it is not needed by `index` because indexing uses the embedding model.
Switching between `openbmb/minicpm5-2b:latest` and `gemma4:e4b` does not require
re-indexing the PDFs.

Thinking is only partially wired at present. Ollama's chat request is made at
[`assistant.py:53`](/Users/dzui_/Documents/Software/Coding/Courses-RAG/src/courses_rag/assistant.py:53)
with `"think": False`, so the app currently asks every model for a final answer
without thinking enabled. The app reads `message.content` and deliberately does
not display or save `message.thinking`. There is no chat command or environment
variable for changing this yet.

MiniCPM's installed Ollama template does contain a thinking branch. A direct
local API request with `think: true` returned separate `message.thinking` and
`message.content` fields; this application currently discards the former. Gemma
4 E2B/E4B also supports an explicit thinking toggle.

For a manual experiment, change that one request field to `"think": True`:

```python
"model": self.model, "stream": False, "think": True,
```

This enables thinking for models whose Ollama template supports it, including
the Gemma 4 E2B/E4B family. It still will not show the reasoning trace in this
app because `message.thinking` is not printed; only the final answer is used.
Thinking tokens also count against `num_predict` in the request, so a small
output limit can end with an empty `message.content` before the final answer is
generated. The app's current `num_predict` is 1200. See the [Ollama chat API](https://docs.ollama.com/api/chat)
and [Ollama thinking guide](https://docs.ollama.com/capabilities/thinking) for the response fields and
model support. Google documents Gemma 4 E2B/E4B thinking as an explicit on/off
mode in its [Gemma thinking guide](https://ai.google.dev/gemma/docs/capabilities/thinking).

## Configuration

```bash
uv run pdf-rag chat --model YOUR_INSTALLED_OLLAMA_MODEL
uv run pdf-rag chat --data /path/to/pdfs --db /path/to/index --topic biology
```

`OLLAMA_MODEL` and `OLLAMA_URL` environment variables override the defaults.
If Ollama reports a connection or model error, check `ollama list` and start the
Ollama application. Credentials are not required for local inference.

If macOS reports `ModuleNotFoundError: courses_rag` after a successful `uv sync`,
check whether its editable-install path file was marked hidden. Python skips
hidden `.pth` files. Clear that flag and retry:

```bash
chflags nohidden .venv/lib/python*/site-packages/courses_rag.pth
```

`python indexer.py [--topic TOPIC | --all]` and
`python query_engine.py [--topic TOPIC]` remain available with the project
dependencies installed. These wrappers import the source directly, so they also
work if macOS hides the editable-install path file. `courses-rag` is an alias for `pdf-rag`.
No source-code edits are needed to choose topics or ask questions.

## Project layout and checks

```text
src/courses_rag/library.py    PDF discovery, extraction, indexing, retrieval
src/courses_rag/assistant.py  Grounded answers, summaries, conversation history
src/courses_rag/cli.py        Commands and interactive chat
indexer.py                   Index command wrapper
query_engine.py              Chat command wrapper
tests/                      Offline regression tests
chroma_db/                  Generated persistent index (gitignored)
.tmp/                       Regenerable model/download caches (gitignored)
```

```bash
uv run python -m unittest discover -s tests -v
```

Tests use a deterministic stand-in embedder and mocked generation, with real
Chroma storage, so they need no model downloads or running Ollama server.

API references: [Ollama chat](https://docs.ollama.com/api/chat) and
[Chroma metadata filtering](https://docs.trychroma.com/docs/querying-collections/metadata-filtering).
