"""Gradio chat UI, used for the hosted demo (e.g. a Hugging Face Space)."""

from __future__ import annotations

from ask_yaron import render
from ask_yaron.agent import Agent

EXAMPLES = [
    "What has Yaron built with AI agents?",
    "Which backend technologies does he use?",
    "Where has he worked since 2024?",
    "What is his education?",
]


def build():
    import gradio as gr

    agent = Agent()

    def respond(message: str, _history: list) -> str:
        return render(agent.ask(message))

    return gr.ChatInterface(
        respond,
        title="Ask my CV: Yaron Lavi",
        description="Answers come only from my CV and project notes, with sources. "
        "If something isn't covered, it says so instead of guessing.",
        examples=EXAMPLES,
    )


def main() -> None:
    build().launch()


if __name__ == "__main__":
    main()
