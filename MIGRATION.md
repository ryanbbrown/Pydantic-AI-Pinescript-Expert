# Migration from pydantic-ai to thinharness

This repository now uses [thinharness](https://github.com/ryanbbrown/thinharness) for its agent loop. The RAG pipeline, database schema, crawler, model configuration values, and structured `PineScriptResult` output remain in place.

## Before and after

| Measure | Before | After | Change |
|---|---:|---:|---:|
| `agent.py` lines | 315 | 309 | 6 fewer lines |
| Direct runtime dependencies | 10 | 10 | Replaced one framework dependency |

The dependency change is `pydantic-ai>=0.0.22` out and `thinharness>=0.5.3` in. The other nine direct runtime dependencies are unchanged. `openai` remains because the retrieval pipeline uses its embeddings client directly.

## Agent-loop changes

- `build_harness()` creates a thinharness `Harness` with one `retrieve` tool and no built-in filesystem tools.
- The retrieval tool receives its database pool and OpenAI embeddings client through a closure. This replaces the former `Dependencies` dataclass and `RunContext` injection.
- Model names are resolved to thinharness provider references at runtime. Values in `config.py` and `OPENROUTER_MODEL` remain raw OpenRouter IDs such as `openai/gpt-4.1-mini`.
- Harnesses are used as asynchronous context managers so their HTTP clients close after use. The interactive shell keeps one harness and database pool open for its session, then closes both on exit.
- `PineScriptResult` remains the structured output type. Callers now read `HarnessResult.output`.

## Behavior fixed during the migration

- Model preset `temperature` and `max_tokens` values are now applied. They were previously defined but ignored.
- `run.py` now reads `result.output` instead of the obsolete `result.data` attribute.
- The interactive shell now passes thinharness resume state between turns. Its former history list was collected but never passed to the agent.
- Streamlit now preserves user and assistant context through thinharness resume state. Its former replay contained only user messages.
- Both Streamlit query paths share and update the same resume state.

## New capabilities and side effects

- Every run is bounded to 8 model requests and 8 tool calls.
- Interactive and Streamlit conversations have real multi-turn memory. The interactive `clear` command resets it.
- thinharness writes local JSON Lines traces to `~/.thinharness/traces/` by default. These traces can include full prompts, model output, and tool payloads. Set `THINHARNESS_DISABLE_LOCAL_TRACING=1`, or configure a harness with `HarnessConfig(local_tracing=False)`, to disable this output.
- Streamlit persists resume state in `chat_resume.json` beside the existing `chat_history.pkl`. The resume file can contain the full transcript and provider reasoning data. Treat both files as sensitive. **Clear Chat History** removes both.

## Intentional non-parity changes

1. Preset temperature and token limits are applied instead of ignored.
2. Interactive and Streamlit conversations use full resume state instead of ineffective or incomplete history handling.
3. `run.py` uses the current result API, `result.output`.
4. Local JSON Lines tracing is enabled by default.
5. Streamlit writes `chat_resume.json` in addition to `chat_history.pkl`.

One behavior is deliberately preserved: if an OpenRouter preset is requested without `OPENROUTER_API_KEY`, the agent logs a warning and falls back to `DEFAULT_MODEL`. Changing this fallback to an error is a separate product decision.

## Verification status

The migration's automated checks cover the harness configuration, retrieval closure, model resolution, existing model output, and unchanged configuration behavior. RAG tests retain their existing live-database skip behavior.

The following live checks are deferred to a developer with API keys and a populated database. They have not been run as part of this migration:

- Single-query CLI calls through the default OpenAI model and each OpenRouter preset.
- A multi-retrieval query to confirm that the 8 model request and 8 tool call limits are suitable.
- Interactive follow-up memory and `clear` against a live model.
- Streamlit visual inspection, follow-up memory, restart persistence, and clear behavior.
- `python api_debug.py`, which makes a live model request.

## Open questions and existing issue

- OpenRouter models differ in structured-output support. If a preset fails with `output_mode="auto"`, test `output_mode="tool"` for that harness and document the exception.
- Resume state grows with the conversation because it contains the transcript. The persisted `chat_resume.json` file grows with it.
- The 8/8 run limits need confirmation with a live multi-retrieval query before they are treated as final tuning values.
- `python run.py populate` imports a missing `populate_db` module. This issue predates the framework migration and was not changed.
