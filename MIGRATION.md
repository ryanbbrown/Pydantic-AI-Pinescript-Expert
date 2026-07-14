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

## Code comparison

### Tool definition and dependency injection

Before, dependencies traveled through a dataclass and `RunContext`:

```python
@dataclass
class Dependencies:
    openai: AsyncOpenAI
    pool: asyncpg.Pool
    openrouter_api_key: str = None   # never read
    use_openrouter: bool = False     # never read

@pinescript_agent.tool
async def retrieve(ctx: RunContext[Dependencies], search_query: str) -> str:
    docs = await hybrid_retrieve(
        pool=ctx.deps.pool, openai_client=ctx.deps.openai, query=search_query, ...
    )
```

After, the tool is a closure over its two real dependencies, registered as a typed `ToolSpec`:

```python
def build_harness(pool: asyncpg.Pool, openai_client: AsyncOpenAI, *, model, temperature, max_tokens) -> Harness:
    async def retrieve(args: RetrieveArgs) -> str:
        docs = await hybrid_retrieve(pool=pool, openai_client=openai_client, query=args.search_query, ...)
        ...

    return Harness(
        HarnessConfig(
            model=model,
            system_prompt=SYSTEM_PROMPT,
            output_type=PineScriptResult,
            builtin_tools=[],            # retrieve is the model's only tool
            max_model_requests=8,        # bounded runs
            max_tool_calls=8,
        ),
        tools=[ToolSpec(name="retrieve", description="...", parameters=RetrieveArgs, handler=retrieve)],
    )
```

### Model routing

Before, OpenRouter routing needed a provider object, a `None` sentinel, and three override branches:

```python
def create_openrouter_model(model_id: str | None = None):
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        return None
    provider = OpenAIProvider(base_url=OPENROUTER_BASE_URL, api_key=openrouter_api_key)
    return OpenAIModel(chosen, provider=provider)

# in run_agent:
if model_override:
    with pinescript_agent.override(model=model_override):
        answer = await pinescript_agent.run(question, deps=deps)
elif use_openrouter:
    or_model = create_openrouter_model()
    if or_model:
        with pinescript_agent.override(model=or_model):
            answer = await pinescript_agent.run(question, deps=deps)
    else:
        answer = await pinescript_agent.run(question, deps=deps)
else:
    answer = await pinescript_agent.run(question, deps=deps)
```

After, routing is a pure function from preset to `provider:model` reference, and the run site has one shape:

```python
model, temperature, max_tokens = resolve_model(preset)
harness = build_harness(pool, openai, model=model, temperature=temperature, max_tokens=max_tokens)
async with harness:
    return await harness.run(question)
```

### Multi-turn conversation

Before, Streamlit replayed only prior *user* messages — the model never saw its own answers:

```python
previous_messages = []
for msg in history:
    if msg["role"] == "user":
        previous_messages.append(ModelRequest(parts=[UserPromptPart(content=msg["content"])]))
result = await pinescript_agent.run(prompt, deps=deps, message_history=previous_messages)
```

After, both UIs continue the actual conversation through thinharness resume state:

```python
result = await harness.run(prompt, resume_from=st.session_state.resume_state)
st.session_state.resume_state = result.resume_state
```

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

The migration's automated checks cover the harness configuration, retrieval closure, model resolution, existing model output, and unchanged configuration behavior (38 tests). RAG tests retain their existing live-database skip behavior.

Live checks run against a pgvector Postgres populated with real embedded documentation chunks and the default OpenAI model:

- `python run.py check` — schema validated, document count reported.
- `python run.py query "How do I create a moving average crossover strategy?"` — full structured answer through the harness.
- Two-turn resume — turn 2 correctly answered a question that required turn 1's content, through the same harness instance via `resume_from`.
- Live retrieval — a documentation-lookup query triggered a real `retrieve` tool call; hybrid search returned the correct document and the answer cited its URL. Confirmed via `tool_call_records` and the local thinharness traces.
- Streamlit — manually exercised: query answered, snippet caption shown, resume state persisted.

Still deferred: OpenRouter preset calls (no live key available during migration testing) and a multi-retrieval query to confirm the 8 model-request / 8 tool-call limits are suitable.

One observation from live testing worth knowing: for common questions, `gpt-4o-mini` frequently answers from its own knowledge without calling `retrieve` — and then self-reports a nonzero `snippets_used`, since that field is filled by the model. This matches the original agent's design (the tool-calling decision was always the model's), but if grounding in the official docs is a hard requirement, consider a system-prompt instruction to always retrieve before answering.

## Open questions and existing issue

- OpenRouter models differ in structured-output support. If a preset fails with `output_mode="auto"`, test `output_mode="tool"` for that harness and document the exception.
- Resume state grows with the conversation because it contains the transcript. The persisted `chat_resume.json` file grows with it.
- The 8/8 run limits need confirmation with a live multi-retrieval query before they are treated as final tuning values.
- `python run.py populate` imports a missing `populate_db` module. This issue predates the framework migration and was not changed.
