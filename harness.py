#!/usr/bin/env python3
"""Ping-pong: Antigravity (agy) and Cursor (agent) implement, then fact-check each other."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CURSOR_MODEL = "cursor-grok-4.6-xhigh-fast"  # Grok 4.6 extra-high effort + fast
AGY_MODEL = "gemini-3.7-flash-high"  # Gemini 3.7 extended thinking (high)
AGY_EFFORT = "high"

REVIEW_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "score", "evaluation", "feedback"],
    "properties": {
        "verdict": {"type": "string", "enum": ["PASS", "FAIL", "REVISE"]},
        "score": {"type": "integer", "minimum": 1, "maximum": 10},
        "evaluation": {"type": "array", "items": {"type": "string"}},
        "feedback": {"type": "array", "items": {"type": "string"}},
    },
}

DEFAULT_TASK = (
    "Evaluate the codebase and the manuscript and ensure all claims, equations, "
    "details, results, figures, etc. in the manuscript match the actual codebase "
    "implementation. Evaluate each claim and edit the codebase and/or manuscript "
    "accordingly to ensure there are 0 fabricated data and false claims. Ensure "
    "the manuscript is concise, clear, and accurate at PNAS/Nature-quality for "
    "submission with correct and logical scientific basis."
)

IMPLEMENT_PREAMBLE = """You are the implementer. Edit the repo (code and/or paper/) to address the task and any critic feedback.
Do not stop at advice: apply the edits. Keep changes minimal and scientifically accurate.
Narrate brief progress as you work (file opened, mismatch found, edit made).
When done, summarize what you changed in a few bullets.
"""

REVIEW_PREAMBLE = """You are the critic. Do not edit files. Fact-check the implementer's work against the task.
Read the actual code and manuscript; do not trust the git diff alone. Flag fabricated numbers, false claims, and equation/code mismatches.
Narrate brief progress as you check claims (file + MATCH/MISMATCH + the number you compared).
Return ONLY JSON with keys:
  verdict: PASS | FAIL | REVISE
  score: integer 1-10
  evaluation: array of strings — claim-by-claim findings with the actual numbers/equations you checked (MATCH or MISMATCH + evidence). This is the review record; never omit it.
  feedback: array of actionable strings (empty if PASS)
PASS only if tests passed and you found no false claims. REVISE if edits are still needed. FAIL if the approach is wrong.
"""


def parse_duration(text: str) -> int:
    """Go-style duration (2h, 30m, 5m0s) → seconds."""
    raw = text.strip().lower()
    total = 0
    for amount, unit in re.findall(r"([0-9.]+)([hms])", raw):
        n = float(amount)
        total += int(n * {"h": 3600, "m": 60, "s": 1}[unit])
    if total == 0:
        total = int(float(raw))
    return total


def _as_str_list(value) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [json.dumps(value, ensure_ascii=False)]
    return [
        json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else str(x)
        for x in value
    ]


def _loads_review_json(text: str) -> tuple[object, str]:
    """Parse JSON; leftover is any narrative before the blob (often the evaluation)."""
    try:
        return json.loads(text), ""
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1)), text[: fenced.start()].strip()
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1]), text[:start].strip()
    raise ValueError(f"critic stdout was not JSON:\n{text[:2000]}")


def extract_stream_result(text: str) -> str | None:
    """Last NDJSON result event payload, or None if this is not a stream."""
    last = None
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and (obj.get("type") == "result" or obj.get("event") == "result"):
            last = obj
    if last is None:
        return None
    return _result_payload(last)


def _result_payload(obj: dict) -> str:
    payload = obj.get("result")
    if isinstance(payload, dict):
        structured = payload.get("structured_output")
        if structured is not None:
            return structured if isinstance(structured, str) else json.dumps(structured)
        if payload.get("response"):
            return str(payload["response"])
        if payload.get("verdict"):
            return json.dumps(payload)
        return json.dumps(payload)
    if isinstance(payload, str):
        return payload
    structured = obj.get("structured_output")
    if structured is not None:
        return structured if isinstance(structured, str) else json.dumps(structured)
    if obj.get("response"):
        return str(obj["response"])
    return ""


def parse_review(stdout: str) -> dict:
    extracted = extract_stream_result(stdout)
    return _parse_review_body(extracted if extracted is not None else stdout)


def _parse_review_body(stdout: str) -> dict:
    text = stdout.strip()
    if not text:
        raise ValueError("empty critic stdout")
    obj, leftover = _loads_review_json(text)
    if isinstance(obj, dict) and "verdict" not in obj:
        inner = obj.get("result") or obj.get("response") or obj.get("data")
        if isinstance(inner, str):
            return _parse_review_body(inner)
        if isinstance(inner, dict):
            obj = inner
    if not isinstance(obj, dict) or obj.get("verdict") not in {"PASS", "FAIL", "REVISE"}:
        raise ValueError(f"invalid review payload: {obj!r}"[:2000])
    evaluation = _as_str_list(obj.get("evaluation"))
    if not evaluation and leftover:
        evaluation = [leftover]
    score = int(obj.get("score") or 0)
    return {
        "verdict": obj["verdict"],
        "score": min(max(score, 1), 10) if score else 1,
        "evaluation": evaluation,
        "feedback": _as_str_list(obj.get("feedback")),
    }


def format_critic_handoff(review: dict) -> str:
    parts = []
    if review.get("evaluation"):
        parts.append(
            "Evaluation results:\n" + "\n".join(f"- {x}" for x in review["evaluation"])
        )
    if review.get("feedback"):
        parts.append("Feedback:\n" + "\n".join(f"- {x}" for x in review["feedback"]))
    return "\n\n".join(parts) or review["verdict"]


def _print_section(title: str, items: list[str]) -> None:
    if not items:
        return
    print(f"    {title}:", flush=True)
    for item in items:
        lines = str(item).splitlines() or [""]
        print(f"      - {lines[0]}", flush=True)
        for line in lines[1:]:
            print(f"        {line}", flush=True)


def _assistant_text(obj: dict) -> str:
    msg = obj.get("message")
    if isinstance(msg, str):
        return msg
    if isinstance(msg, dict):
        parts = []
        for block in msg.get("content") or []:
            if isinstance(block, dict) and block.get("text"):
                parts.append(str(block["text"]))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(obj.get("text") or obj.get("text_delta") or "")


def _brief_args(args: dict, limit: int = 140) -> str:
    if not args:
        return ""
    prefer = (
        "path",
        "Path",
        "target_file",
        "file_path",
        "command",
        "CommandLine",
        "query",
        "pattern",
        "glob",
    )
    bits = [f"{k}={args[k]}" for k in prefer if k in args]
    if not bits:
        bits = [f"{k}={v}" for k, v in list(args.items())[:2]]
    text = " ".join(str(b) for b in bits)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _cursor_tool(tool_call: dict) -> tuple[str, dict]:
    fn = tool_call.get("function")
    if isinstance(fn, dict):
        raw = fn.get("arguments")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                raw = {"arguments": raw}
        return str(fn.get("name") or "function"), raw if isinstance(raw, dict) else {}
    for key, val in tool_call.items():
        name = key[: -len("ToolCall")] if key.endswith("ToolCall") else key
        args = val.get("args") if isinstance(val, dict) else {}
        return name, args if isinstance(args, dict) else {}
    return "tool", {}


class _Progress:
    """Print live NDJSON from Cursor and agy; keep the terminal result payload."""

    def __init__(self) -> None:
        self.saw_partial = False
        self.in_text = False
        self.result: str | None = None
        self.lines: list[str] = []

    def _break(self) -> None:
        if self.in_text:
            print(flush=True)
            self.in_text = False

    def _write(self, text: str) -> None:
        if not text:
            return
        sys.stdout.write(text)
        sys.stdout.flush()
        self.in_text = not text.endswith("\n")

    def feed(self, line: str) -> None:
        self.lines.append(line)
        raw = line.strip()
        if not raw:
            return
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            self._break()
            print(line, end="", flush=True)
            return
        if not isinstance(obj, dict):
            return
        kind = obj.get("type") or obj.get("event")
        if kind in {"system", "init"}:
            model = obj.get("model") or (obj.get("init") or {}).get("model") or ""
            self._break()
            print(f"    [init] {model}".rstrip(), flush=True)
        elif kind == "assistant":
            if obj.get("timestamp_ms"):
                self.saw_partial = True
            if obj.get("model_call_id"):
                return
            if self.saw_partial and not obj.get("timestamp_ms"):
                return
            self._write(_assistant_text(obj))
        elif kind == "tool_call":
            name, args = _cursor_tool(obj.get("tool_call") or {})
            self._break()
            if obj.get("subtype") == "started":
                print(f"    [{name}] {_brief_args(args)}".rstrip(), flush=True)
            else:
                print(f"    [{name} done]", flush=True)
        elif kind == "step_update":
            step = obj.get("step_update") or {}
            stype = step.get("step_type") or ""
            state = step.get("state") or ""
            if stype == "agent_response" and step.get("text_delta"):
                self._write(str(step["text_delta"]))
                return
            if stype == "tool":
                info = step.get("tool_info") or {}
                name = step.get("tool_name") or info.get("name") or "tool"
                extra = _brief_args(info.get("parameters") or {})
                self._break()
                tag = f"    [{name}] {extra}".rstrip() if state == "ACTIVE" else f"    [{name} done] {extra}".rstrip()
                print(tag, flush=True)
                return
            if state == "ACTIVE" and stype and stype not in {"user_input", "checkpoint"}:
                self._break()
                print(f"    [{stype}]", flush=True)
        elif kind == "result":
            self._break()
            self.result = _result_payload(obj)
            inner = obj["result"] if isinstance(obj.get("result"), dict) else obj
            dur = None
            if isinstance(inner, dict) and inner.get("duration_seconds") is not None:
                dur = float(inner["duration_seconds"])
            elif obj.get("duration_ms") is not None:
                dur = float(obj["duration_ms"]) / 1000.0
            extra = f" {dur:.0f}s" if dur is not None else ""
            print(f"    [done{extra}]", flush=True)


def _which(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"{name} not on PATH. Install the CLI, then retry.")
    return path


def run_logged(argv: list[str], *, timeout: int) -> subprocess.CompletedProcess:
    print("$", " ".join(argv[:8]), "…" if len(argv) > 8 else "", flush=True)
    proc = subprocess.Popen(
        argv,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None and proc.stderr is not None
    progress = _Progress()

    def read_out() -> None:
        for line in proc.stdout:
            progress.feed(line)

    def read_err() -> None:
        for line in proc.stderr:
            sys.stderr.write(line)
            sys.stderr.flush()

    t_out = threading.Thread(target=read_out, daemon=True)
    t_err = threading.Thread(target=read_err, daemon=True)
    t_out.start()
    t_err.start()
    try:
        rc = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        proc.kill()
        proc.wait()
        t_out.join(5)
        t_err.join(5)
        raise TimeoutError(f"{argv[0]} timed out after {timeout}s") from exc
    t_out.join(10)
    t_err.join(10)
    stdout = progress.result if progress.result is not None else "".join(progress.lines)
    if rc != 0:
        tail = "".join(progress.lines)[-4000:]
        raise RuntimeError(f"{argv[0]} failed (exit {rc}):\n{tail}")
    return subprocess.CompletedProcess(argv, rc, stdout, "")


def run_antigravity(prompt: str, *, review: bool, print_timeout: str) -> str:
    argv = [
        _which("agy"),
        "--model",
        AGY_MODEL,
        "--effort",
        AGY_EFFORT,
        "--mode",
        "plan" if review else "accept-edits",
        "--dangerously-skip-permissions",
        "--print-timeout",
        print_timeout,
        "--output-format",
        "stream-json",
    ]
    if review:
        schema = Path(tempfile.gettempdir()) / "pi-nsde-review-schema.json"
        schema.write_text(json.dumps(REVIEW_SCHEMA), encoding="utf-8")
        argv.extend(["--json-schema", str(schema)])
    argv.extend(["-p", prompt])
    return run_logged(argv, timeout=parse_duration(print_timeout) + 120).stdout


def run_cursor(prompt: str, *, review: bool, print_timeout: str) -> str:
    argv = [
        _which("agent"),
        "--model",
        CURSOR_MODEL,
        "--output-format",
        "stream-json",
        "--stream-partial-output",
        "--trust",
        "--workspace",
        str(ROOT),
        "--sandbox",
        "disabled",
    ]
    argv.extend(["--mode", "ask"] if review else ["--force"])
    argv.extend(["-p", prompt])
    return run_logged(argv, timeout=parse_duration(print_timeout) + 120).stdout


def run_pytest() -> subprocess.CompletedProcess:
    if shutil.which("uv"):
        cmd = ["uv", "run", "pytest", "-q"]
    else:
        cmd = [sys.executable, "-m", "pytest", "-q"]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


def git_diff(limit: int = 80_000) -> str:
    proc = subprocess.run(["git", "diff"], cwd=ROOT, capture_output=True, text=True)
    diff = proc.stdout or ""
    if len(diff) > limit:
        diff = diff[:limit] + f"\n… truncated ({len(diff)} chars)"
    return diff


def implement(who: str, task: str, feedback: str, print_timeout: str) -> None:
    prompt = (
        f"{IMPLEMENT_PREAMBLE}\nTask:\n{task}\n\n"
        f"Previous critic feedback:\n{feedback or '(none — first pass)'}\n"
    )
    print(f"\n>>> {who} implements", flush=True)
    if who == "antigravity":
        run_antigravity(prompt, review=False, print_timeout=print_timeout)
    else:
        run_cursor(prompt, review=False, print_timeout=print_timeout)


def critique(who: str, task: str, tests: str, diff: str, print_timeout: str) -> dict:
    prompt = (
        f"{REVIEW_PREAMBLE}\nTask:\n{task}\n\nGit diff:\n{diff or '(no diff)'}\n\n"
        f"Pytest:\n{tests}\n"
    )
    print(f"\n>>> {who} reviews", flush=True)
    if who == "antigravity":
        stdout = run_antigravity(prompt, review=True, print_timeout=print_timeout)
    else:
        stdout = run_cursor(prompt, review=True, print_timeout=print_timeout)
    review = parse_review(stdout)
    print(f"    verdict={review['verdict']}  score={review['score']}", flush=True)
    _print_section("evaluation", review["evaluation"])
    _print_section("feedback", review["feedback"])
    print(json.dumps(review, ensure_ascii=False, indent=2), flush=True)
    return review


def run_cycle(task: str, max_turns: int, print_timeout: str, starter: str) -> int:
    pair = ("antigravity", "cursor") if starter == "antigravity" else ("cursor", "antigravity")
    feedback = ""
    last = None
    for turn in range(max_turns):
        implementer = pair[turn % 2]
        critic = pair[(turn + 1) % 2]
        print(f"\n=== turn {turn + 1}/{max_turns}: {implementer} → {critic} ===", flush=True)
        implement(implementer, task, feedback, print_timeout)
        tests = run_pytest()
        test_txt = f"exit={tests.returncode}\n{tests.stdout}\n{tests.stderr}"
        print(tests.stdout or tests.stderr or f"pytest exit {tests.returncode}", flush=True)
        last = critique(critic, task, test_txt, git_diff(), print_timeout)
        if last["verdict"] == "PASS" and tests.returncode == 0:
            print(f"\nConverged on turn {turn + 1}.")
            return 0
        feedback = format_critic_handoff(last)
        if tests.returncode != 0:
            feedback += f"\nPytest failed:\n{tests.stdout}\n{tests.stderr}"
    print(f"\nStopped after {max_turns} turns without PASS.")
    return 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", nargs="?", default=DEFAULT_TASK)
    p.add_argument("--max-turns", type=int, default=4)
    p.add_argument(
        "--print-timeout",
        default="2h",
        help="agy --print-timeout (Go duration). Default 2h; agy default is 5m and will abort long runs.",
    )
    p.add_argument("--starter", choices=("antigravity", "cursor"), default="antigravity")
    args = p.parse_args(argv)
    return run_cycle(args.task, args.max_turns, args.print_timeout, args.starter)


if __name__ == "__main__":
    raise SystemExit(main())
