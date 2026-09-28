"""Start a PDF conversation; run `python query_engine.py --help` for options."""
from src.courses_rag.cli import main

if __name__ == "__main__":
    import sys
    main(["chat", *sys.argv[1:]])
