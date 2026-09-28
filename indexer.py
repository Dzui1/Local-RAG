"""Index PDFs; run `python indexer.py --help` for options."""
from src.courses_rag.cli import main

if __name__ == "__main__":
    import sys
    main(["index", *sys.argv[1:]])
