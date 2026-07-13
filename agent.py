from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import asyncpg
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from thinharness import Harness, HarnessConfig, ToolSpec

from config import (
    DEFAULT_DATABASE_URL,
    DEFAULT_MODEL,
    EMBEDDING_MODEL,
    HYBRID_SEARCH_ALPHA,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    MMR_LAMBDA,
    MODEL_PRESETS,
    OPENROUTER_DEFAULT_MODEL,
    RERANK_TOP_N,
    RETRIEVAL_CANDIDATES,
    SIMILARITY_THRESHOLD,
    get_preset,
)
from rag_utils import hybrid_retrieve

# Force reload environment variables
load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Log masked API keys at DEBUG level for startup diagnostics
openai_key = os.getenv("OPENAI_API_KEY", "")
openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
logger.debug(
    "OpenAI API key loaded: %s...%s",
    openai_key[:4],
    openai_key[-4:] if len(openai_key) > 8 else "",
)
logger.debug(
    "OpenRouter API key loaded: %s...%s",
    openrouter_key[:4],
    openrouter_key[-4:] if len(openrouter_key) > 8 else "",
)


def get_openai_api_key() -> str:
    """Get OpenAI API key with validation and user prompt if needed."""
    # Try to get from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Check if key is missing or has a placeholder value
    if not openai_api_key or openai_api_key in ["YOUR_OPENAI_API_KEY", "sk-...", "YOUR_OPE***_API"] or "YOUR_" in openai_api_key:
        logger.warning("OPENAI_API_KEY is not set or has a placeholder value.")
        print("Please enter your OpenAI API key:")
        openai_api_key = input("> ")

        if not openai_api_key:
            raise ValueError("OpenAI API key is required to continue.")

        # Save to environment for this session
        os.environ["OPENAI_API_KEY"] = openai_api_key

        # Also update the .env file for future runs
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        try:
            from dotenv import set_key
            set_key(env_path, "OPENAI_API_KEY", openai_api_key)
            logger.info("Updated OPENAI_API_KEY in %s for future runs", env_path)
        except ImportError:
            logger.warning("Could not update .env file - dotenv.set_key not available")
            logger.warning("Please manually update %s with your API key", env_path)

    # Verify it's not still using a placeholder
    if "YOUR_" in openai_api_key or openai_api_key == "sk-...":
        logger.warning("API key appears to be a placeholder value.")
        print("Please enter your actual OpenAI API key:")
        openai_api_key = input("> ")

        if not openai_api_key:
            raise ValueError("OpenAI API key is required to continue.")

        # Save to environment for this session
        os.environ["OPENAI_API_KEY"] = openai_api_key

    return openai_api_key


# Define result structure for better tracking
class PineScriptResult(BaseModel):
    query: str = Field(description="The original query")
    response: str = Field(description="The generated response")
    snippets_used: int = Field(description="Number of documentation snippets used")


class RetrieveArgs(BaseModel):
    search_query: str = Field(
        description="The search query to find relevant documentation"
    )


SYSTEM_PROMPT = (
    "You are a Pine Script v6 expert assistant. Pine Script is the programming language used in TradingView "
    "for creating custom indicators and strategies for technical analysis of financial markets. "
    "Your task is to provide clear, accurate information about Pine Script v6 based on the official documentation. "
    "Always include code examples in your explanations when relevant. "
    "Focus on being practical and giving working solutions for user problems.\n\n"
    "When answering questions about Pine Script, follow these guidelines:\n"
    "1. Include working code examples whenever possible\n"
    "2. Explain each part of the code clearly\n"
    "3. Highlight any common pitfalls or best practices\n"
    "4. If you're unsure about something, be transparent about it\n"
    "5. Format your code with proper syntax highlighting\n"
    "6. When appropriate, mention TradingView-specific context\n"
    "7. Reference specific Pine Script v6 functions and features accurately\n"
    "8. Provide clear explanations of complex concepts with analogies when helpful\n"
)


def _model_ref(value: str) -> str:
    """Prefix raw OpenRouter IDs while preserving provider-qualified refs."""
    slash_index = value.find("/")
    colon_index = value.find(":")
    if colon_index >= 0 and (slash_index < 0 or colon_index < slash_index):
        return value
    return f"openrouter:{value}"


def resolve_model(preset: str | None) -> tuple[str, float, int]:
    """Map a preset name, raw OpenRouter ID, or None to model settings."""
    if preset and preset in MODEL_PRESETS:
        cfg = get_preset(preset)
        model = _model_ref(str(cfg["model"]))
        temperature = float(cfg["temperature"])
        max_tokens = int(cfg["max_tokens"])
        logger.info("Using preset '%s' → %s", preset, cfg["model"])
    elif preset:
        model = _model_ref(preset)
        temperature = LLM_TEMPERATURE
        max_tokens = LLM_MAX_TOKENS
        logger.info("Using raw model ID: %s", preset)
    elif os.getenv("OPENROUTER_API_KEY"):
        model = _model_ref(OPENROUTER_DEFAULT_MODEL)
        temperature = LLM_TEMPERATURE
        max_tokens = LLM_MAX_TOKENS
        logger.info("Using OpenRouter default model")
    else:
        logger.info("Using default OpenAI model")
        return DEFAULT_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS

    if model.startswith("openrouter:") and not os.getenv("OPENROUTER_API_KEY"):
        logger.warning("OPENROUTER_API_KEY not found, using default OpenAI model")
        return DEFAULT_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS

    return model, temperature, max_tokens


def build_harness(
    pool: asyncpg.Pool,
    openai_client: AsyncOpenAI,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
) -> Harness:
    """Build a Pine Script agent harness around the retrieval dependencies."""

    async def retrieve(args: RetrieveArgs) -> str:
        logger.debug("Running hybrid retrieval for: %s", args.search_query)
        docs = await hybrid_retrieve(
            pool=pool,
            openai_client=openai_client,
            query=args.search_query,
            embedding_model=EMBEDDING_MODEL,
            candidates=RETRIEVAL_CANDIDATES,
            alpha=HYBRID_SEARCH_ALPHA,
            threshold=SIMILARITY_THRESHOLD,
            top_n=RERANK_TOP_N,
            mmr_lambda=MMR_LAMBDA,
        )

        if not docs:
            return (
                "No relevant documentation found in the database. "
                "The database may need to be populated with Pine Script documentation."
            )

        logger.debug("Hybrid retrieval returned %d documents", len(docs))
        return "\n\n".join(
            f"# {doc.title}\nDocumentation URL: {doc.url}\n\n{doc.content}\n"
            for doc in docs
        )

    return Harness(
        HarnessConfig(
            root=".",
            model=model,
            system_prompt=SYSTEM_PROMPT,
            temperature=temperature,
            max_tokens=max_tokens,
            output_type=PineScriptResult,
            output_mode="auto",
            builtin_tools=[],
            max_model_requests=8,
            max_tool_calls=8,
        ),
        tools=[
            ToolSpec(
                name="retrieve",
                description=(
                    "Retrieve relevant Pine Script documentation using the hybrid search "
                    "pipeline: vector and BM25 search, RRF fusion, similarity filtering, "
                    "cross-encoder reranking, and MMR deduplication."
                ),
                parameters=RetrieveArgs,
                handler=retrieve,
            )
        ],
    )


@asynccontextmanager
async def database_connect(create_db: bool = False) -> AsyncGenerator[asyncpg.Pool, None]:
    """Connect to the database with the Pine Script documentation.

    Args:
        create_db: Whether to create the database if it doesn't exist

    Yields:
        asyncpg.Pool: A connection pool to the database
    """
    # Use the connection string directly from environment variables
    db_url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)

    logger.info("Connecting to database: %s", db_url)
    try:
        pool = await asyncpg.create_pool(db_url)
        logger.info("Database connection established")
        try:
            yield pool
        finally:
            await pool.close()
            logger.info("Database connection closed")
    except Exception as e:
        logger.error("Error connecting to database: %s", e)
        raise


async def run_agent(question: str, preset: str | None = None):
    """Run the agent with a specific question.

    Args:
        question: The user's Pine Script question.
        preset:   Optional model preset name (``codex``, ``opus``, ``flash``)
                  or a raw OpenRouter model ID (e.g. ``anthropic/claude-opus-4-6``).
                  When omitted the agent falls back to OpenRouter default or OpenAI.
    """
    logger.info("Running agent with question: %s", question)

    openai_api_key = get_openai_api_key()
    model, temperature, max_tokens = resolve_model(preset)

    try:
        async with database_connect(False) as pool:
            async with AsyncOpenAI(api_key=openai_api_key) as openai:
                harness = build_harness(
                    pool,
                    openai,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                async with harness:
                    return await harness.run(question)
    except Exception as e:
        logger.error("Error running agent: %s", e)
        return None


async def main():
    """Main function to run the agent from the command line."""
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "How do I define a variable in Pine Script v6?"

    print(f"Question: {question}")
    result = await run_agent(question)

    if result:
        print("\nResponse:")
        print(result.output.response)
        print(f"\nSnippets used: {result.output.snippets_used}")
    else:
        print("No response received from the agent.")


if __name__ == "__main__":
    asyncio.run(main())
