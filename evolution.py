"""
evolution.py - Continuous Evolution & Self-Learning Engine
Captures interactions, user feedback, new coding patterns, and consolidates them
into permanent episodic memory and continual fine-tuning data.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional


class EvolutionEngine:
    def __init__(self, memory_dir: str = "memory"):
        self.memory_dir = memory_dir
        os.makedirs(memory_dir, exist_ok=True)
        self.log_file = os.path.join(memory_dir, "interactions.jsonl")
        self.profile_file = os.path.join(memory_dir, "profile.json")
        self.notes_file = os.path.join(memory_dir, "evolution_notes.md")
        self.profile = self._load_profile()

    def _load_profile(self) -> Dict:
        if os.path.exists(self.profile_file):
            try:
                with open(self.profile_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "created_at": datetime.now().isoformat(),
            "total_interactions": 0,
            "learned_facts": [],
            "user_preferences": {},
            "coding_habits": {}
        }

    def _save_profile(self):
        with open(self.profile_file, "w", encoding="utf-8") as f:
            json.dump(self.profile, f, indent=2)

    def log_interaction(self, user_prompt: str, bot_response: str, user_feedback: Optional[str] = None):
        """Record every conversation turn so Kira never forgets"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt": user_prompt,
            "response": bot_response,
            "feedback": user_feedback
        }
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        self.profile["total_interactions"] += 1
        self._save_profile()

    def teach_fact(self, fact_category: str, key: str, value: str):
        """Explicitly teach Kira a new rule, coding convention, or personal preference"""
        if fact_category not in self.profile:
            self.profile[fact_category] = {}
        self.profile[fact_category][key] = value
        self.profile["learned_facts"].append({
            "category": fact_category,
            "key": key,
            "value": value,
            "timestamp": datetime.now().isoformat()
        })
        self._save_profile()
        print(f"Kira learned: [{fact_category}] {key} = {value}")

    def recall(self, query: str, k: int = 3) -> List[Dict]:
        """Search taught facts and past interactions (BM25, local). Returns [{text, kind, score}]."""
        from retrieval import BM25Index

        idx = BM25Index()
        for fact in self.profile.get("learned_facts", []):
            idx.add({"text": f"{fact['category']} {fact['key']}: {fact['value']}", "kind": "fact"})
        if os.path.exists(self.log_file):
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        e = json.loads(line)
                    except ValueError:
                        continue
                    answer = e.get("feedback") or e.get("response") or ""
                    if answer.startswith("From what I have indexed"):
                        continue  # don't recall our own retrieval dumps
                    idx.add({"text": f"User: {e.get('prompt', '')}\nKira: {answer}", "kind": "interaction"})
        return idx.search(query, k=k)

    def consolidate_memory(self):
        """Summarize all recent experiences into evolution notes"""
        if not os.path.exists(self.log_file):
            return "No interactions logged yet."

        interactions = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    interactions.append(json.loads(line))

        summary_md = f"# Kira Evolution Log\n*Last consolidated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        summary_md += f"- **Total Conversations:** {len(interactions)}\n"
        summary_md += f"- **Known Facts:** {len(self.profile.get('learned_facts', []))}\n\n"
        summary_md += "## Active User Preferences & Conventions\n"
        for k, v in self.profile.get("user_preferences", {}).items():
            summary_md += f"- **{k}**: {v}\n"

        summary_md += "\n## Recent Learning History\n"
        for entry in interactions[-10:]:
            summary_md += f"- *{entry['timestamp'][:19]}*: Prompt: `{entry['prompt'][:50]}...`\n"

        with open(self.notes_file, "w", encoding="utf-8") as f:
            f.write(summary_md)

        print(f"Memory consolidated! Saved to {self.notes_file}")
        return summary_md

    def export_for_continual_training(self, output_path: str = "memory/continual_train.jsonl"):
        """Export high-value interactions and learned facts into training pairs for LoRA/fine-tuning"""
        pairs = []
        if os.path.exists(self.log_file):
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    # If user corrected the model, prioritize that learning
                    prompt = data["prompt"]
                    resp = data["feedback"] if data.get("feedback") else data["response"]
                    pairs.append({"prompt": prompt, "response": resp})

        # Add explicit learned facts
        for fact in self.profile.get("learned_facts", []):
            pairs.append({
                "prompt": f"What is the {fact['category']} for {fact['key']}?",
                "response": str(fact["value"])
            })

        with open(output_path, "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")

        print(f"Exported {len(pairs)} continual learning samples to {output_path}")
        return len(pairs)
