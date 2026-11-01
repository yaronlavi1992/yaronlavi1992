"""Vercel function: POST {"question": "..."} and get a grounded answer back."""

from __future__ import annotations

import json
import re
import sys
import traceback
from http.server import BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ask_yaron.agent import Agent  # noqa: E402

MAX_BODY_BYTES = 4000
BUSY = "The demo is busy right now. Please try again in a minute."
AGENT: Agent | None = None


def answer_payload(question: str, agent: Agent) -> dict:
    answer = agent.ask(question)
    titles = {c.id: c.title for c in agent.chunks}
    text = re.sub(r"\s*\[[a-z0-9-]+\]", "", answer.text).strip()
    return {
        "answer": text,
        "sources": [titles[i] for i in answer.citations if i in titles],
        "refused": answer.refused,
    }


class handler(BaseHTTPRequestHandler):  # noqa: N801 - Vercel looks for this exact name
    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        global AGENT
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= MAX_BODY_BYTES:
            self._send(400, {"error": "Please send a shorter question."})
            return
        try:
            question = str(json.loads(self.rfile.read(length)).get("question", ""))
        except (ValueError, AttributeError):
            self._send(400, {"error": 'Send JSON like {"question": "..."}.'})
            return
        try:
            AGENT = AGENT or Agent()
            self._send(200, answer_payload(question, AGENT))
        except Exception:
            traceback.print_exc()
            self._send(503, {"error": BUSY})
