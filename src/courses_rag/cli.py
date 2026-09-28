import argparse

from .assistant import Assistant
from .library import Library


HELP = """/topics                 List topic folders
/use TOPIC              Switch topic (keeps each topic's conversation)
/files                  List PDFs in this topic
/pdf FILENAME           Focus on one PDF (exact filename, no quotes needed)
/all                    Use all PDFs in this topic
/summary                Summarize the selected PDF(s)
/index                  Refresh this topic after changing PDFs
/reset                  Clear this topic's conversation
/help                   Show commands
/quit                   Exit"""


def chat(library, assistant, topic):
    library.index(topic)
    selected = {}
    print("PDF assistant — '.' is the default topic (PDFs directly in data/).\n" + HELP)
    while True:
        source = selected.get(topic)
        try:
            question = input(f"\n[{topic} | {source or 'all PDFs'}] > ").strip()
            if not question:
                continue
            command, _, argument = question.partition(" ")
            argument = argument.strip()
            if command == "/quit":
                return
            if command == "/help":
                print(HELP)
            elif command == "/topics":
                show_topics(library)
            elif command == "/use":
                library.index(argument)
                topic = argument
            elif command == "/files":
                print("\n".join(p.name for p in library.files(topic)))
            elif command == "/pdf":
                library.records(topic, argument or "(missing filename)")
                selected[topic] = argument
                assistant.histories.pop(topic, None)
            elif command == "/all":
                selected.pop(topic, None)
                assistant.histories.pop(topic, None)
            elif command == "/reset":
                assistant.histories.pop(topic, None)
            elif command == "/index":
                library.index(topic)
                assistant.histories.pop(topic, None)
                if source and source not in {p.name for p in library.files(topic)}:
                    selected.pop(topic, None)
            elif command == "/summary":
                print(assistant.ask(topic, argument or "Summarize each PDF's key ideas.", source, summary=True))
            elif command.startswith("/"):
                print("Unknown command. Type /help.")
            else:
                print(assistant.ask(topic, question, source))
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return
        except (ValueError, RuntimeError) as exc:
            print(f"Error: {exc}")


def show_topics(library):
    topics = library.topics()
    if not topics:
        print(f"No PDFs found in {library.data}. Add PDFs here or in topic subfolders.")
    for topic, files in topics.items():
        print(f"{topic}: {len(files)} PDF(s)")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Ask questions about PDFs, organized by topic folder.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("topics", "index", "ask", "chat"):
        sub = commands.add_parser(name)
        sub.add_argument("--data", default="data", help="PDF root directory (default: data)")
        sub.add_argument("--db", default="chroma_db", help="Persistent index directory")
        if name != "topics":
            sub.add_argument("--topic", help="Folder relative to data; defaults to '.' or the first available topic")
        if name == "index":
            sub.add_argument("--all", action="store_true", help="Index every topic folder")
        if name in ("ask", "chat"):
            sub.add_argument("--model", help="Ollama model; also configurable with OLLAMA_MODEL")
        if name == "ask":
            sub.add_argument("question")
            sub.add_argument("--pdf", help="Exact PDF filename within the topic")
            sub.add_argument("--summary", action="store_true", help="Read the complete selected PDF(s)")
    args = parser.parse_args(argv)
    library = Library(args.data, args.db)
    try:
        if args.command != "topics" and args.topic is None:
            topics = library.topics()
            if not topics:
                raise ValueError(f"No PDFs found in {library.data}. Add PDFs here or in topic subfolders.")
            args.topic = "." if "." in topics else next(iter(topics))
            if args.topic != ".":
                print(f"Selected topic: {args.topic}")
        if args.command == "topics":
            show_topics(library)
        elif args.command == "index":
            for topic in library.topics() if args.all else [args.topic]:
                print(f"[{topic}]")
                library.index(topic)
        else:
            assistant = Assistant(library, args.model)
            if args.command == "chat":
                chat(library, assistant, args.topic)
            else:
                library.index(args.topic)
                print(assistant.ask(args.topic, args.question, args.pdf, args.summary))
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
