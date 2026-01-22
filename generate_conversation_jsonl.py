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
import os
import time
import requests
from pathlib import Path
from typing import Optional

# OpenRouter API configuration
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "gryphe/mythomax-l2-13b"

# olivia's key 
OPENROUTER_API_KEY = "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"


def call_openrouter(
    user_content: str,
    system_prompt: str = None,
    model: str = DEFAULT_MODEL,
    api_key: str = None,
    max_tokens: int = 2048,
    temperature: float = 0.7
) -> str:
    """
    Call OpenRouter API to generate a response.

    Args:
        user_content: The user message/question
        system_prompt: Optional system prompt to set model behavior
        model: Model ID to use (default: dolphin-mistral-24b-venice-edition:free)
        api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY constant)
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature

    Returns:
        Generated response text
    """
    api_key = api_key or OPENROUTER_API_KEY or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OpenRouter API key required. Set OPENROUTER_API_KEY or pass api_key parameter.")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/moral-reasoning-steering",
        "X-Title": "Moral Reasoning Dataset Generator"
    }

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload)
    response.raise_for_status()

    result = response.json()
    return result["choices"][0]["message"]["content"]


def generate_with_llm(
    csv_file: str,
    output_file: str,
    system_column: str = "system_prompt",
    question_column: str = "question",
    model: str = DEFAULT_MODEL,
    api_key: str = None,
    delay: float = 1.0,
    start_index: int = 0,
    limit: int = None
) -> int:
    """
    Generate JSONL by calling OpenRouter API for each row in CSV.

    Args:
        csv_file: Path to input CSV with system prompts and questions
        output_file: Path to output JSONL file
        system_column: CSV column name for system prompts
        question_column: CSV column name for user questions
        model: OpenRouter model ID
        api_key: OpenRouter API key
        delay: Delay between API calls (seconds) to avoid rate limits
        start_index: Row index to start from (for resuming)
        limit: Maximum number of rows to process

    Returns:
        Number of conversations generated
    """
    count = 0
    mode = 'a' if start_index > 0 else 'w'

    with open(csv_file, 'r', encoding='utf-8') as csv_f:
        reader = list(csv.DictReader(csv_f))
        end_index = len(reader) if limit is None else min(len(reader), start_index + limit)
        total = end_index - start_index

        with open(output_file, mode, encoding='utf-8') as jsonl_f:
            for i, row in enumerate(reader):
                if i < start_index:
                    continue
                if i >= end_index:
                    break

                system_prompt = row.get(system_column, "")
                question = row[question_column]

                print(f"[{i - start_index + 1}/{total}] Generating response...")

                try:
                    response = call_openrouter(
                        user_content=question,
                        system_prompt=system_prompt,
                        model=model,
                        api_key=api_key
                    )

                    # Determine action from response (1 if takes action, 0 if refuses/inaction)
                    action = determine_action(response)

                    conversation = create_conversation(question, response, action)
                    jsonl_f.write(json.dumps(conversation, ensure_ascii=False) + '\n')
                    jsonl_f.flush()  # Flush after each write for safety

                    count += 1
                    print(f"    Action: {action} ({'action' if action == 1 else 'inaction'})")

                    if delay > 0 and i < end_index - 1:
                        time.sleep(delay)

                except Exception as e:
                    print(f"    ERROR: {e}")
                    continue

    return count


def determine_action(response: str) -> int:
    """
    Determine if response indicates taking action (1) or inaction (0).

    Looks for common patterns indicating the model's decision.
    """
    response_lower = response.lower()
    first_500 = response_lower[:500]

    # Patterns indicating inaction (refusing to take the action)
    inaction_patterns = [
        "i do not", "i don't", "i would not", "i wouldn't",
        "i refuse", "i cannot", "i can't", "i will not", "i won't",
        "no, i", "no. i", "answer: no", "answer is no"
    ]

    # Patterns indicating action (taking the action)
    action_patterns = [
        "i would", "i will", "i do", "yes, i", "yes. i",
        "i take", "i choose to", "i decide to", "i proceed",
        "answer: yes", "answer is yes"
    ]

    # Check first part of response for decision
    for pattern in inaction_patterns:
        if pattern in first_500:
            return 0

    for pattern in action_patterns:
        if pattern in first_500:
            return 1

    # Default to 1 if unclear (can be manually corrected)
    return 1


def create_conversation(user_content: str, assistant_content: str, action: int = None) -> dict:
    """Create a single conversation object.

    Args:
        user_content: The user's message content
        assistant_content: The assistant's response content
        action: Optional action indicator (0 = inaction/not taking action, 1 = taking action)
    """
    assistant_message = {"role": "assistant", "content": assistant_content}
    if action is not None:
        assistant_message["action"] = action

    return {
        "messages": [
            {"role": "user", "content": user_content},
            assistant_message
        ]
    }


def add_conversation(output_file: str, user_content: str, assistant_content: str, action: int = None) -> None:
    """Append a single conversation to a JSONL file.

    Args:
        output_file: Path to output JSONL file
        user_content: The user's message content
        assistant_content: The assistant's response content
        action: Optional action indicator (0 = inaction/not taking action, 1 = taking action)
    """
    conversation = create_conversation(user_content, assistant_content, action)
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(conversation, ensure_ascii=False) + '\n')


def generate_from_list(output_file: str, conversations: list[tuple], overwrite: bool = True) -> int:
    """
    Generate JSONL from a list of conversation tuples.

    Args:
        output_file: Path to output JSONL file
        conversations: List of tuples - either (user_message, assistant_message) or
                      (user_message, assistant_message, action) where action is 0 or 1
        overwrite: If True, overwrite existing file; if False, append

    Returns:
        Number of conversations written
    """
    mode = 'w' if overwrite else 'a'
    with open(output_file, mode, encoding='utf-8') as f:
        for conv in conversations:
            if len(conv) == 3:
                user_content, assistant_content, action = conv
            else:
                user_content, assistant_content = conv
                action = None
            conversation = create_conversation(user_content, assistant_content, action)
            f.write(json.dumps(conversation, ensure_ascii=False) + '\n')
    return len(conversations)


def generate_from_csv(csv_file: str, output_file: str,
                      user_column: str = 'user',
                      assistant_column: str = 'assistant',
                      action_column: str = None) -> int:
    """
    Generate JSONL from a CSV file.

    Args:
        csv_file: Path to input CSV file
        output_file: Path to output JSONL file
        user_column: Name of column containing user messages
        assistant_column: Name of column containing assistant messages
        action_column: Optional name of column containing action indicator (0 or 1)

    Returns:
        Number of conversations written
    """
    count = 0
    with open(csv_file, 'r', encoding='utf-8') as csv_f, \
         open(output_file, 'w', encoding='utf-8') as jsonl_f:
        reader = csv.DictReader(csv_f)
        for row in reader:
            action = None
            if action_column and action_column in row:
                action = int(row[action_column])
            conversation = create_conversation(
                row[user_column],
                row[assistant_column],
                action
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

        print("Action indicator (0 = inaction, 1 = action, or press Enter to skip):")
        action_input = input().strip()
        action = int(action_input) if action_input in ('0', '1') else None

        add_conversation(output_file, user_content, assistant_content, action)
        count += 1
        print(f"Added conversation {count}\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate conversation JSONL files')
    parser.add_argument('output', help='Output JSONL file path')
    parser.add_argument('--csv', help='Input CSV file')
    parser.add_argument('--user-col', default='user', help='CSV column for user messages')
    parser.add_argument('--assistant-col', default='assistant', help='CSV column for assistant messages')
    parser.add_argument('--action-col', default=None, help='CSV column for action indicator (0=inaction, 1=action)')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')

    # LLM generation options
    parser.add_argument('--generate', '-g', action='store_true',
                        help='Generate responses using OpenRouter LLM')
    parser.add_argument('--system-col', default='system_prompt',
                        help='CSV column for system prompts (used with --generate)')
    parser.add_argument('--question-col', default='question',
                        help='CSV column for questions (used with --generate)')
    parser.add_argument('--model', default=DEFAULT_MODEL,
                        help=f'OpenRouter model ID (default: {DEFAULT_MODEL})')
    parser.add_argument('--api-key', default=None,
                        help='OpenRouter API key (overrides built-in key)')
    parser.add_argument('--delay', type=float, default=1.0,
                        help='Delay between API calls in seconds (default: 1.0)')
    parser.add_argument('--start', type=int, default=0,
                        help='Row index to start from (for resuming)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Maximum number of rows to process')

    args = parser.parse_args()

    if args.generate and args.csv:
        # LLM generation mode
        print(f"Generating responses using {args.model}")
        print(f"Input: {args.csv}")
        print(f"Output: {args.output}")
        if args.start > 0:
            print(f"Resuming from row {args.start}")
        if args.limit:
            print(f"Limiting to {args.limit} rows")
        print()

        count = generate_with_llm(
            csv_file=args.csv,
            output_file=args.output,
            system_column=args.system_col,
            question_column=args.question_col,
            model=args.model,
            api_key=args.api_key,
            delay=args.delay,
            start_index=args.start,
            limit=args.limit
        )
        print(f"\nGenerated {count} conversations to {args.output}")

    elif args.csv:
        count = generate_from_csv(args.csv, args.output, args.user_col, args.assistant_col, args.action_col)
        print(f"Generated {count} conversations from {args.csv} to {args.output}")
    elif args.interactive:
        interactive_mode(args.output)
    else:
        print("Example usage:")
        print()
        print("1. Generate responses with LLM:")
        print(f"   python3 {__file__} output.jsonl --csv questions.csv --generate")
        print(f"   python3 {__file__} output.jsonl --csv questions.csv --generate --limit 5")
        print()
        print("2. Interactive mode:")
        print(f"   python3 {__file__} output.jsonl --interactive")
        print()
        print("3. From CSV (with pre-existing responses):")
        print(f"   python3 {__file__} output.jsonl --csv input.csv")
