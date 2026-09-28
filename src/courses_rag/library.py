"""PDF extraction, topic discovery, and persistent vector indexing."""
import hashlib
from pathlib import Path


def chunk_text(text: str, chunk_size: int = 160, overlap: int = 30) -> list[str]:
    if not 0 <= overlap < chunk_size:
        raise ValueError("Require 0 <= overlap < chunk_size")
    words = text.split()
    return [" ".join(words[i:i + chunk_size])
            for i in range(0, len(words), chunk_size - overlap)]


class Library:
    def __init__(self, data="data", database="chroma_db", embedder=None):
        self.data = Path(data).resolve()
        self.database = str(database)
        self._client = None
        self._embedder = embedder

    def topics(self):
        """Each folder containing PDFs is a topic; '.' means data itself."""
        result = {}
        for path in sorted(self.data.rglob("*")):
            if path.is_file() and path.suffix.lower() == ".pdf":
                if not path.resolve().is_relative_to(self.data):
                    continue
                topic = path.parent.relative_to(self.data).as_posix()
                result.setdefault(topic, []).append(path)
        return result

    def files(self, topic):
        topics = self.topics()
        if topic not in topics:
            raise ValueError(f"Unknown or empty topic: {topic}. Use 'topics' to list PDF folders.")
        return topics[topic]

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu",
                                                cache_folder=".tmp/huggingface/hub")
        return self._embedder

    def collection(self, topic):
        import chromadb
        if self._client is None:
            self._client = chromadb.PersistentClient(path=self.database)
        identity = f"{self.data}\n{topic}"
        name = "pdf-" + hashlib.sha256(identity.encode()).hexdigest()[:32]
        return self._client.get_or_create_collection(name=name, embedding_function=None)

    def index(self, topic):
        from pypdf import PdfReader
        files = self.files(topic)
        collection = self.collection(topic)
        existing = collection.get(include=["metadatas"])["metadatas"] or []
        fingerprints = {}
        counts = {}
        expected = {}
        for meta in existing:
            fingerprints.setdefault(meta["source"], set()).add(meta.get("fingerprint"))
            counts[meta["source"]] = counts.get(meta["source"], 0) + 1
            expected[meta["source"]] = meta.get("total_chunks")
        current = {p.name for p in files}
        for removed in fingerprints.keys() - current:
            collection.delete(where={"source": removed})
            print(f"  Removed: {removed}")
        for path in files:
            fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
            if (fingerprints.get(path.name) == {fingerprint}
                    and counts[path.name] == expected[path.name]):
                print(f"  Unchanged: {path.name}")
                continue
            documents, metadatas = [], []
            reader = PdfReader(path)
            for page_number, page in enumerate(reader.pages, 1):
                for chunk_number, chunk in enumerate(chunk_text(page.extract_text() or "")):
                    documents.append(chunk)
                    metadatas.append(dict(source=path.name, page=page_number,
                                          chunk=chunk_number, fingerprint=fingerprint))
            if not documents:
                collection.delete(where={"source": path.name})
                print(f"  No extractable text: {path.name} (scanned PDFs need OCR)")
                continue
            # Complete extraction and embedding before replacing a previous version.
            embeddings = self.embedder.encode(documents).tolist()
            for meta in metadatas:
                meta["total_chunks"] = len(documents)
            collection.delete(where={"source": path.name})
            prefix = hashlib.sha256(path.name.encode()).hexdigest()
            for start in range(0, len(documents), 128):
                end = min(start + 128, len(documents))
                collection.add(ids=[f"{prefix}-{i}" for i in range(start, end)],
                               documents=documents[start:end], metadatas=metadatas[start:end],
                               embeddings=embeddings[start:end])
            print(f"  Indexed: {path.name} ({len(reader.pages)} pages, {len(documents)} chunks)")
        return collection

    def records(self, topic, source=None):
        files = self.files(topic)
        if source and source not in {p.name for p in files}:
            raise ValueError(f"PDF not found in topic {topic}: {source}")
        options = {"where": {"source": source}} if source else {}
        result = self.collection(topic).get(include=["documents", "metadatas"], **options)
        records = list(zip(result["documents"], result["metadatas"]))
        if not records:
            raise ValueError("No indexed text. Run index first; scanned PDFs require OCR.")
        return sorted(records, key=lambda r: (r[1]["source"], r[1]["page"], r[1]["chunk"]))

    def search(self, topic, question, source=None):
        records = self.records(topic, source)
        options = {"where": {"source": source}} if source else {}
        result = self.collection(topic).query(
            query_embeddings=[self.embedder.encode(question).tolist()],
            n_results=min(6, len(records)), **options)
        return list(zip(result["documents"][0], result["metadatas"][0]))
