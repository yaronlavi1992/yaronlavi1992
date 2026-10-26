"""Ask my CV: a grounded question-answering agent over Yaron Lavi's CV."""

from __future__ import annotations

import argparse
import sys

from ask_yaron.agent import Agent, Answer

__all__ = ["Agent", "Answer", "main"]


def render(answer: Answer) -> str:
    if answer.citations:
        return f"{answer.text}\n\nSources: {', '.join(answer.citations)}"
    return answer.text


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions about Yaron Lavi's experience.")
    parser.add_argument("question", nargs="*", help="question to ask; omit for interactive mode")
    args = parser.parse_args()
    agent = Agent()
    if args.question:
        print(render(agent.ask(" ".join(args.question))))
        return
    print("Ask about Yaron's experience (Ctrl-D to quit).")
    for line in sys.stdin:
        if line.strip():
            print(render(agent.ask(line)), end="\n\n")
