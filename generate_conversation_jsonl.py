#!/usr/bin/env python3
"""
Generate JSONL files in the conversation format:
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Usage:
    1. Programmatic: Import and use add_conversation() or generate_from_list()
    2. Interactive: Run script directly to add conversations one at a time
    3. From CSV: Use generate_from_csv() with a CSV file containing 'user' and 'assistant' columns
"""

import json
import csv
from pathlib import Path
from typing import Optional


def create_conversation(user_content: str, assistant_content: str) -> dict:
    """Create a single conversation object."""
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
    }


def add_conversation(output_file: str, user_content: str, assistant_content: str) -> None:
    """Append a single conversation to a JSONL file."""
    conversation = create_conversation(user_content, assistant_content)
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(conversation, ensure_ascii=False) + '\n')


def generate_from_list(output_file: str, conversations: list[tuple[str, str]], overwrite: bool = True) -> int:
    """
    Generate JSONL from a list of (user_content, assistant_content) tuples.

    Args:
        output_file: Path to output JSONL file
        conversations: List of (user_message, assistant_message) tuples
        overwrite: If True, overwrite existing file; if False, append

    Returns:
        Number of conversations written
    """
    mode = 'w' if overwrite else 'a'
    with open(output_file, mode, encoding='utf-8') as f:
        for user_content, assistant_content in conversations:
            conversation = create_conversation(user_content, assistant_content)
            f.write(json.dumps(conversation, ensure_ascii=False) + '\n')
    return len(conversations)


def generate_from_csv(csv_file: str, output_file: str,
                      user_column: str = 'user',
                      assistant_column: str = 'assistant') -> int:
    """
    Generate JSONL from a CSV file.

    Args:
        csv_file: Path to input CSV file
        output_file: Path to output JSONL file
        user_column: Name of column containing user messages
        assistant_column: Name of column containing assistant messages

    Returns:
        Number of conversations written
    """
    count = 0
    with open(csv_file, 'r', encoding='utf-8') as csv_f, \
         open(output_file, 'w', encoding='utf-8') as jsonl_f:
        reader = csv.DictReader(csv_f)
        for row in reader:
            conversation = create_conversation(
                row[user_column],
                row[assistant_column]
            )
            jsonl_f.write(json.dumps(conversation, ensure_ascii=False) + '\n')
            count += 1
    return count


def read_jsonl(input_file: str) -> list[dict]:
    """Read and parse a JSONL file."""
    conversations = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                conversations.append(json.loads(line))
    return conversations


def interactive_mode(output_file: str) -> None:
    """Run interactive mode to add conversations one at a time."""
    print(f"Interactive mode - conversations will be appended to: {output_file}")
    print("Enter 'quit' or 'q' to exit\n")

    count = 0
    while True:
        print(f"--- Conversation {count + 1} ---")

        print("User message (enter 'quit' to exit, or type message then press Enter twice):")
        user_lines = []
        while True:
            line = input()
            if line.lower() in ('quit', 'q') and not user_lines:
                print(f"\nExiting. Added {count} conversation(s) to {output_file}")
                return
            if line == '' and user_lines:
                break
            user_lines.append(line)
        user_content = '\n'.join(user_lines)

        print("Assistant message (type message then press Enter twice):")
        assistant_lines = []
        while True:
            line = input()
            if line == '' and assistant_lines:
                break
            assistant_lines.append(line)
        assistant_content = '\n'.join(assistant_lines)

        add_conversation(output_file, user_content, assistant_content)
        count += 1
        print(f"Added conversation {count}\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate conversation JSONL files')
    parser.add_argument('output', help='Output JSONL file path')
    parser.add_argument('--csv', help='Input CSV file (requires --user-col and --assistant-col)')
    parser.add_argument('--user-col', default='user', help='CSV column for user messages')
    parser.add_argument('--assistant-col', default='assistant', help='CSV column for assistant messages')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')

    args = parser.parse_args()

    if args.csv:
        count = generate_from_csv(args.csv, args.output, args.user_col, args.assistant_col)
        print(f"Generated {count} conversations from {args.csv} to {args.output}")
    elif args.interactive:
        interactive_mode(args.output)
    else:
        # Demo mode - show example usage
        print("Example usage:")
        print()
        print("1. Interactive mode:")
        print(f"   python {__file__} output.jsonl --interactive")
        print()
        print("2. From CSV:")
        print(f"   python {__file__} output.jsonl --csv input.csv")
        print()
        print("3. Programmatic (in Python):")
        print("""
   from generate_conversation_jsonl import generate_from_list, add_conversation

   # Generate from list of tuples
   conversations = [
       ("What is Python?", "Python is a programming language..."),
       ("How do I read a file?", "You can use open() to read files..."),
   ]
   generate_from_list("output.jsonl", conversations)

   # Or add one at a time
   add_conversation("output.jsonl", "Hello", "Hi there!")
""")
