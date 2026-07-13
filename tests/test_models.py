"""Tests for PineScriptResult model validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent import PineScriptResult


# ---------------------------------------------------------------------------
# PineScriptResult — validation
# ---------------------------------------------------------------------------

class TestPineScriptResult:
    def test_valid_result(self) -> None:
        result = PineScriptResult(
            query="How do I plot a line?",
            response="Use the plot() function...",
            snippets_used=3,
        )
        assert result.query == "How do I plot a line?"
        assert result.response == "Use the plot() function..."
        assert result.snippets_used == 3

    def test_zero_snippets(self) -> None:
        result = PineScriptResult(
            query="test", response="answer", snippets_used=0
        )
        assert result.snippets_used == 0

    def test_missing_query_raises(self) -> None:
        with pytest.raises(ValidationError):
            PineScriptResult(response="answer", snippets_used=1)  # type: ignore[call-arg]

    def test_missing_response_raises(self) -> None:
        with pytest.raises(ValidationError):
            PineScriptResult(query="q", snippets_used=1)  # type: ignore[call-arg]

    def test_missing_snippets_used_raises(self) -> None:
        with pytest.raises(ValidationError):
            PineScriptResult(query="q", response="r")  # type: ignore[call-arg]

    def test_snippets_must_be_int(self) -> None:
        with pytest.raises(ValidationError):
            PineScriptResult(query="q", response="r", snippets_used="many")  # type: ignore[arg-type]

    def test_empty_strings_allowed(self) -> None:
        result = PineScriptResult(query="", response="", snippets_used=0)
        assert result.query == ""

    def test_long_response(self) -> None:
        long_text = "x" * 50_000
        result = PineScriptResult(
            query="test", response=long_text, snippets_used=1
        )
        assert len(result.response) == 50_000

    def test_serialisation_roundtrip(self) -> None:
        original = PineScriptResult(
            query="plot question", response="use plot()", snippets_used=5
        )
        data = original.model_dump()
        restored = PineScriptResult(**data)
        assert restored == original

    def test_json_roundtrip(self) -> None:
        original = PineScriptResult(
            query="test", response="answer", snippets_used=2
        )
        json_str = original.model_dump_json()
        restored = PineScriptResult.model_validate_json(json_str)
        assert restored == original
