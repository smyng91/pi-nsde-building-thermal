"""Harness review JSON parsing — no live CLI calls."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import harness  # noqa: E402


def test_parse_duration():
    assert harness.parse_duration("2h") == 7200
    assert harness.parse_duration("5m0s") == 300
    assert harness.parse_duration("90") == 90


def test_parse_review_cursor_envelope():
    payload = {"verdict": "REVISE", "score": 6, "feedback": ["fix C"]}
    envelope = json.dumps({"type": "result", "result": json.dumps(payload)})
    review = harness.parse_review(envelope)
    assert review["verdict"] == "REVISE"
    assert review["feedback"] == ["fix C"]
    assert review["evaluation"] == []


def test_parse_review_fenced_and_nested():
    raw = 'Here you go:\n```json\n{"verdict":"PASS","score":9,"feedback":[]}\n```\n'
    review = harness.parse_review(raw)
    assert review["verdict"] == "PASS"
    assert review["evaluation"] == ["Here you go:"]
    nested = json.dumps({"result": {"verdict": "FAIL", "score": 2, "feedback": "no"}})
    review = harness.parse_review(nested)
    assert review["verdict"] == "FAIL"
    assert review["feedback"] == ["no"]


def test_parse_review_keeps_evaluation():
    payload = {
        "verdict": "REVISE",
        "score": 6,
        "evaluation": ["UnknownC=5.409 MATCH fig2 seed0"],
        "feedback": ["fix C"],
    }
    review = harness.parse_review(json.dumps(payload))
    assert review["evaluation"] == ["UnknownC=5.409 MATCH fig2 seed0"]
    assert "UnknownC=5.409" in harness.format_critic_handoff(review)


def test_parse_review_prefix_is_evaluation_fallback():
    raw = 'Checked Table 1 C=5.409.\n{"verdict":"PASS","score":9,"feedback":[]}'
    review = harness.parse_review(raw)
    assert review["evaluation"] == ["Checked Table 1 C=5.409."]


def test_parse_review_stream_json_cursor_and_agy():
    payload = {"verdict": "PASS", "score": 9, "evaluation": ["ok"], "feedback": []}
    cursor = "\n".join(
        [
            json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "checking"}]}}),
            json.dumps({"type": "result", "subtype": "success", "result": json.dumps(payload)}),
        ]
    )
    assert harness.parse_review(cursor)["verdict"] == "PASS"
    agy = json.dumps(
        {
            "event": "result",
            "result": {
                "status": "SUCCESS",
                "response": "{}",
                "structured_output": {
                    "verdict": "REVISE",
                    "score": 5,
                    "evaluation": ["C"],
                    "feedback": ["fix"],
                },
            },
        }
    )
    review = harness.parse_review(agy)
    assert review["verdict"] == "REVISE"
    assert review["evaluation"] == ["C"]


def test_progress_prints_tools_and_text(capsys):
    progress = harness._Progress()
    progress.feed(
        json.dumps(
            {
                "type": "tool_call",
                "subtype": "started",
                "tool_call": {"readToolCall": {"args": {"path": "paper/main.tex"}}},
            }
        )
        + "\n"
    )
    progress.feed(
        json.dumps(
            {
                "type": "assistant",
                "timestamp_ms": 1,
                "message": {"content": [{"type": "text", "text": "checking C"}]},
            }
        )
        + "\n"
    )
    progress.feed(
        json.dumps(
            {
                "event": "step_update",
                "step_update": {
                    "step_type": "tool",
                    "state": "ACTIVE",
                    "tool_name": "read_file",
                    "tool_info": {"name": "read_file", "parameters": {"Path": "sde.py"}},
                },
            }
        )
        + "\n"
    )
    progress.feed(
        json.dumps(
            {
                "event": "result",
                "result": {"status": "SUCCESS", "response": "done", "duration_seconds": 3},
            }
        )
        + "\n"
    )
    out = capsys.readouterr().out
    assert "paper/main.tex" in out
    assert "checking C" in out
    assert "sde.py" in out
    assert "[done 3s]" in out
    assert progress.result == "done"
