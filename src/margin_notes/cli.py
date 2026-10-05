import argparse
import json
from pathlib import Path

from .answer import answer
from .index import Index, MiniLM
from .models import Document, Query


def main():
    parser = argparse.ArgumentParser(description="Ingest and search local documents")
    parser.add_argument("--db", default="data/notes.sqlite")
    parser.add_argument("--semantic", action="store_true", help="Load the pinned MiniLM model")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest")
    ingest.add_argument("file", type=Path, help="JSONL documents")
    query = commands.add_parser("query")
    query.add_argument("question")
    query.add_argument("--collection", default="demo")
    delete = commands.add_parser("delete")
    delete.add_argument("document_id")
    delete.add_argument("--collection", default="demo")
    commands.add_parser("stats")
    args = parser.parse_args()
    index = Index(args.db, MiniLM() if args.semantic else None)
    if args.command == "ingest":
        # Validate the whole input first so malformed JSON cannot cause partial ingest.
        documents = [
            Document.model_validate_json(line)
            for line in args.file.read_text().splitlines()
            if line.strip()
        ]
        changed = sum(index.ingest(doc) for doc in documents)
        print(json.dumps({"changed": changed, **index.stats()}))
    elif args.command == "query":
        print(
            answer(
                index,
                Query(
                    question=args.question,
                    collection=args.collection,
                    mode="hybrid" if args.semantic else "lexical",
                ),
            ).model_dump_json(indent=2)
        )
    elif args.command == "delete":
        index.delete(args.collection, args.document_id)
        print(json.dumps(index.stats()))
    else:
        print(json.dumps(index.stats()))


if __name__ == "__main__":
    main()
