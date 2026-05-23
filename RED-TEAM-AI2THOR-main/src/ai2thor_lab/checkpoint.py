"""Durable checkpoint for adversarial attack runs.

Lets a benchmark session survive a full process restart — kill the server,
disconnect Unity, reboot the machine — and pick up where it left off when
the user hits "Start attack" again, unless they tick "Clear previous
results" first.

Two artifacts on disk (under `<project_root>/results/`):

  verdicts.jsonl       — per-case judge verdicts, appended incrementally
                         as they happen. Source of truth for "which cases
                         have I already run?".
  strategy_memory.json — UCB1 counters, successful prompts, rejection
                         reasons, leaderboard. Lets the red agent's
                         cross-case learning carry over between runs.

The resume key is (instruction, strategy, batch_label). Two attempts with
the same instruction under different strategies or different batch labels
are considered different cases and are both run.
"""

from __future__ import annotations

import json
import os
from typing import Iterable, Optional


# ── Paths ──────────────────────────────────────────────────────────────────

def verdicts_path(project_root: str) -> str:
    return os.path.join(project_root, "results", "verdicts.jsonl")


def memory_path(project_root: str) -> str:
    return os.path.join(project_root, "results", "strategy_memory.json")


# ── Resume key ─────────────────────────────────────────────────────────────

def case_key(instruction: str, strategy: str, batch_label: str) -> tuple:
    """Canonical key used to skip already-completed cases on resume."""
    return (
        (instruction or "").strip(),
        (strategy or "").strip(),
        (batch_label or "").strip(),
    )


def load_completed_keys(project_root: str) -> set:
    """Scan verdicts.jsonl and return the set of (instruction, strategy,
    batch_label) tuples that have already produced a verdict.

    Returns an empty set if the file doesn't exist or every line fails to
    parse — callers treat that as "fresh run, nothing to skip".
    """
    path = verdicts_path(project_root)
    seen: set = set()
    if not os.path.exists(path):
        return seen
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    v = json.loads(line)
                except json.JSONDecodeError:
                    continue
                seen.add(case_key(
                    v.get("instruction", ""),
                    v.get("strategy", ""),
                    v.get("batch_label", ""),
                ))
    except OSError:
        pass
    return seen


def load_prior_verdicts(project_root: str) -> list:
    """Return every parsed verdict dict from verdicts.jsonl, in order.

    Used to rehydrate `state.attack_verdicts` at server start so the UI,
    RASE section, and per-strategy tables all reflect prior progress
    before the user hits "Start attack" to resume.
    """
    path = verdicts_path(project_root)
    out: list = []
    if not os.path.exists(path):
        return out
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return out


# ── Strategy memory ────────────────────────────────────────────────────────

def save_strategy_memory(memory, project_root: str) -> None:
    """Persist StrategyMemory to disk atomically.

    Writes to a temp file + rename so a crash mid-write can't corrupt the
    checkpoint.
    """
    try:
        path = memory_path(project_root)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        data = memory.to_dict() if hasattr(memory, "to_dict") else {}
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except Exception as e:
        # Checkpoint failures must never crash the attack loop.
        print(f"[CHECKPOINT] save_strategy_memory failed: {e}")


def load_strategy_memory(project_root: str, memory_cls):
    """Rehydrate a StrategyMemory from disk, or return a fresh one.

    Pass the StrategyMemory class as `memory_cls` so this module does not
    need to import adaptive_red_agent (which pulls heavy deps).
    """
    path = memory_path(project_root)
    if not os.path.exists(path):
        return memory_cls()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if hasattr(memory_cls, "from_dict"):
            return memory_cls.from_dict(data)
    except Exception as e:
        print(f"[CHECKPOINT] load_strategy_memory failed, starting fresh: {e}")
    return memory_cls()


# ── Clear ──────────────────────────────────────────────────────────────────

def clear_all(project_root: str) -> None:
    """Wipe all checkpoint artifacts. Called when the user ticks
    "Clear previous results" in the UI."""
    for p in (verdicts_path(project_root), memory_path(project_root)):
        try:
            if os.path.exists(p):
                os.remove(p)
                print(f"[CHECKPOINT] Removed {p}")
        except OSError as e:
            print(f"[CHECKPOINT] Failed to remove {p}: {e}")


# ── Reporting helper ───────────────────────────────────────────────────────

def summarize_completed(completed_keys: Iterable[tuple]) -> str:
    """Return a short human-readable summary for log messages."""
    n = len(list(completed_keys)) if not isinstance(completed_keys, (set, list)) else len(completed_keys)
    return f"{n} case(s) previously completed — will be skipped"
