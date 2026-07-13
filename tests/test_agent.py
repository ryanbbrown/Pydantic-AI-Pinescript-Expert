"""Tests for the thinharness agent factory and model resolution."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from thinharness import HarnessConfig, ToolSpec

import agent
from agent import PineScriptResult, RetrieveArgs, build_harness, resolve_model
from config import (
    DEFAULT_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    MODEL_PRESETS,
    OPENROUTER_DEFAULT_MODEL,
)
from rag_utils import RetrievedDoc


def _make_dependencies(rows=None):
    """Build mocked OpenAI and asyncpg dependencies for retrieval."""
    embedding_obj = MagicMock()
    embedding_obj.data = [MagicMock(embedding=[0.1] * 1536)]

    openai_mock = AsyncMock()
    openai_mock.embeddings.create = AsyncMock(return_value=embedding_obj)

    pool_mock = AsyncMock()
    pool_mock.fetch = AsyncMock(return_value=rows or [])
    return pool_mock, openai_mock


def _capture_harness(monkeypatch, **build_kwargs):
    """Build through the public factory while capturing its constructor inputs."""
    harness = MagicMock()
    harness_factory = MagicMock(return_value=harness)
    monkeypatch.setattr(agent, "Harness", harness_factory)
    pool, openai_client = _make_dependencies()

    result = build_harness(pool, openai_client, **build_kwargs)

    assert result is harness
    harness_factory.assert_called_once()
    config = harness_factory.call_args.args[0]
    tools = harness_factory.call_args.kwargs["tools"]
    return pool, openai_client, config, tools


class TestRetrieveTool:
    @pytest.mark.asyncio
    async def test_returns_formatted_docs(self, monkeypatch) -> None:
        pool, openai_client, _, tools = _capture_harness(monkeypatch)
        docs = [
            RetrievedDoc(
                url="https://docs.tv/plot",
                title="plot()",
                content="Plots a line on the chart.",
            ),
            RetrievedDoc(
                url="https://docs.tv/hline",
                title="hline()",
                content="Draws a horizontal line.",
            ),
        ]
        retrieve_mock = AsyncMock(return_value=docs)
        monkeypatch.setattr(agent, "hybrid_retrieve", retrieve_mock)

        result = await tools[0].handler(RetrieveArgs(search_query="how to plot a line"))

        assert "# plot()" in result
        assert "# hline()" in result
        assert "Plots a line" in result
        retrieve_mock.assert_awaited_once()
        call_kwargs = retrieve_mock.call_args.kwargs
        assert call_kwargs["pool"] is pool
        assert call_kwargs["openai_client"] is openai_client
        assert call_kwargs["query"] == "how to plot a line"

    @pytest.mark.asyncio
    async def test_returns_message_when_no_docs(self, monkeypatch) -> None:
        _, _, _, tools = _capture_harness(monkeypatch)
        monkeypatch.setattr(agent, "hybrid_retrieve", AsyncMock(return_value=[]))

        result = await tools[0].handler(RetrieveArgs(search_query="nonexistent topic"))

        assert result == (
            "No relevant documentation found in the database. "
            "The database may need to be populated with Pine Script documentation."
        )

    @pytest.mark.asyncio
    async def test_handles_embedding_error(self, monkeypatch) -> None:
        pool, openai_client, _, tools = _capture_harness(monkeypatch)
        openai_client.embeddings.create = AsyncMock(side_effect=RuntimeError("API down"))

        with pytest.raises(RuntimeError, match="API down"):
            await tools[0].handler(RetrieveArgs(search_query="test"))

        pool.fetch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_handles_db_error(self, monkeypatch) -> None:
        pool, _, _, tools = _capture_harness(monkeypatch)
        pool.fetch = AsyncMock(side_effect=ConnectionError("connection refused"))

        with pytest.raises(ConnectionError, match="connection refused"):
            await tools[0].handler(RetrieveArgs(search_query="test"))


class TestAgentWiring:
    def test_factory_passes_one_retrieve_tool(self, monkeypatch) -> None:
        _, _, _, tools = _capture_harness(monkeypatch)

        assert len(tools) == 1
        assert isinstance(tools[0], ToolSpec)
        assert tools[0].name == "retrieve"
        assert tools[0].parameters is RetrieveArgs

    def test_factory_configures_structured_rag_harness(self, monkeypatch) -> None:
        _, _, config, _ = _capture_harness(
            monkeypatch,
            model="openrouter:example/model",
            temperature=0.35,
            max_tokens=3456,
        )

        assert isinstance(config, HarnessConfig)
        assert config.model == "openrouter:example/model"
        assert config.temperature == 0.35
        assert config.max_tokens == 3456
        assert config.output_type is PineScriptResult
        assert config.output_mode == "auto"
        assert config.builtin_tools == []
        assert config.max_model_requests == 8
        assert config.max_tool_calls == 8


class TestResolveModel:
    def test_known_preset_uses_its_model_and_settings(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        preset = MODEL_PRESETS["codex"]

        result = resolve_model("codex")

        assert result == (
            f"openrouter:{preset['model']}",
            preset["temperature"],
            preset["max_tokens"],
        )

    def test_raw_model_id_is_treated_as_openrouter_model(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

        result = resolve_model("vendor/custom-model")

        assert result == (
            "openrouter:vendor/custom-model",
            LLM_TEMPERATURE,
            LLM_MAX_TOKENS,
        )

    @pytest.mark.parametrize(
        "model_ref",
        [
            "openai:gpt-4o-mini",
            "anthropic:claude-sonnet-4",
            "openrouter:vendor/custom-model",
        ],
    )
    def test_already_prefixed_model_passes_through(
        self, monkeypatch, model_ref
    ) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

        result = resolve_model(model_ref)

        assert result == (
            model_ref,
            LLM_TEMPERATURE,
            LLM_MAX_TOKENS,
        )

    def test_no_preset_with_openrouter_key_uses_openrouter_default(
        self, monkeypatch
    ) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

        result = resolve_model(None)

        assert result == (
            f"openrouter:{OPENROUTER_DEFAULT_MODEL}",
            LLM_TEMPERATURE,
            LLM_MAX_TOKENS,
        )

    def test_no_preset_without_openrouter_key_uses_default(self, monkeypatch) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        assert resolve_model(None) == (
            DEFAULT_MODEL,
            LLM_TEMPERATURE,
            LLM_MAX_TOKENS,
        )

    def test_preset_without_openrouter_key_falls_back_to_default(
        self, monkeypatch, caplog
    ) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        result = resolve_model("codex")

        assert result == (DEFAULT_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS)
        assert "OPENROUTER_API_KEY not found" in caplog.text
