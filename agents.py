"""
agents.py - Small multi-agent orchestration: planner -> researcher(s) -> writer.

Each agent is a role with its own prompt over any backend (extractive/Ollama/Kira).
With the default extractive backend the pipeline still runs deterministically: the
planner splits the task into sub-questions by rule, researchers retrieve, the writer
assembles. A real LLM backend just makes each step smarter.
"""

import re
from typing import Dict, List

from backends import ExtractiveBackend


class Agent:
    def __init__(self, name: str, backend, instructions: str):
        self.name, self.backend, self.instructions = name, backend, instructions

    def run(self, task: str) -> str:
        return self.backend.generate(f"{self.instructions}\n\nTask: {task}")


def split_questions(task: str, limit: int = 4) -> List[str]:
    parts = re.split(r"\?\s+|\band then\b|\bthen\b|;|\n|\band\b(?=\s+(?:what|how|why|when|where|who))", task, flags=re.I)
    parts = [p.strip(" .?") for p in parts if p and len(p.strip(" .?")) > 3]
    return parts[:limit] or [task.strip()]


class Orchestrator:
    """Runs a task through planner, researchers (retrieval-backed assistant) and a writer."""

    def __init__(self, assistant, planner_backend=None, writer_backend=None):
        self.assistant = assistant
        self.planner = Agent("planner", planner_backend or ExtractiveBackend(),
                             "Break the task into at most 4 short numbered questions.")
        self.writer = Agent("writer", writer_backend or assistant.backend,
                            "Write a concise final answer from the findings only.")
        self.use_llm_planner = planner_backend is not None and not isinstance(planner_backend, ExtractiveBackend)

    def plan(self, task: str) -> List[str]:
        if self.use_llm_planner:
            try:
                lines = self.planner.run(task).splitlines()
                qs = [re.sub(r"^\s*(\d+[.)]|[-*])\s*", "", ln).strip() for ln in lines]
                qs = [q for q in qs if len(q) > 3][:4]
                if qs:
                    return qs
            except Exception:
                pass
        return split_questions(task)

    def run(self, task: str) -> Dict:
        steps = []
        for q in self.plan(task):
            res = self.assistant.ask(q)
            steps.append({"question": q, "answer": res["answer"], "sources": res["sources"]})
        findings = "\n\n".join(f"Q: {s['question']}\nA: {s['answer']}" for s in steps)
        if isinstance(self.writer.backend, ExtractiveBackend):
            final = "\n\n".join(f"{i}. {s['question']}\n{s['answer']}" for i, s in enumerate(steps, 1))
        else:
            try:
                final = self.writer.run(f"{task}\n\nFindings:\n{findings}")
            except Exception:
                final = findings
        return {"plan": [s["question"] for s in steps], "steps": steps, "final": final}
