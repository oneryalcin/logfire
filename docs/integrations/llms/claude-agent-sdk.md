---
title: "Logfire Integrations: Claude Agent SDK"
description: "Guide for using Logfire with the Claude Agent SDK, including setup instructions and example trace output."
integration: logfire
---
# Claude Agent SDK

You can instrument the Python [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) using **Logfire**.

!!! note
    This is separate from the [`anthropic` integration](../llms/anthropic.md). The Claude Agent SDK doesn't actually use the `anthropic` package under the hood.

First, install dependencies:

```bash
pip install logfire claude-agent-sdk
```

Here's an example script:

```python skip-run="true" skip-reason="external-connection"
import asyncio
from typing import Any

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    create_sdk_mcp_server,
    tool,
)

import logfire

logfire.configure()
logfire.instrument_claude_agent_sdk()


# Example of using a tool in the Claude Agent SDK:
@tool(
    'get_weather',
    'Gets the current weather for a given city',
    {
        'city': str,
    },
)
async def get_weather(args: dict[str, Any]) -> dict[str, Any]:
    """Simulated weather lookup tool"""
    city = args['city']
    weather = 'Cloudy, 59°F'
    return {'content': [{'type': 'text', 'text': f'Weather in {city}: {weather}'}]}


async def main():
    weather_server = create_sdk_mcp_server(
        name='weather',
        version='1.0.0',
        tools=[get_weather],
    )

    options = ClaudeAgentOptions(
        system_prompt='You are a friendly travel assistant who helps with weather information.',
        mcp_servers={'weather': weather_server},
        allowed_tools=['mcp__weather__get_weather'],
    )

    async with ClaudeSDKClient(options=options) as client:
        await client.query("What's the weather like in Berlin?")

        async for message in client.receive_response():
            print(message)


asyncio.run(main())
```

!!! warning
    Only the `ClaudeSDKClient` is instrumented, not the top-level `claude_agent_sdk.query()` function. Instances created **after** calling `logfire.instrument_claude_agent_sdk()` are fully instrumented. Existing instances will get conversation and turn spans but not tool call spans.

## Server-side tools

Server-side tools — `web_search`, `web_fetch`, `code_execution`, `bash_code_execution`, `text_editor_code_execution`, `advisor`, `tool_search_tool_regex`, `tool_search_tool_bm25` — are executed by the Anthropic API on the model's behalf and appear alongside client-side tool calls in the assistant message stream. Logfire captures them as `tool_call` / `tool_call_response` parts under `gen_ai.output.messages`, distinguished by the `name` field.

Server-side tools do **not** trigger the `PreToolUse` / `PostToolUse` hooks (those fire for client-side tools only), so no child `execute_tool` spans are emitted for them — their invocations are visible only as parts of the `chat` span's output messages.

!!! note
    The Claude Agent SDK currently only parses `advisor_tool_result` blocks into typed `ServerToolResultBlock` objects; results for other server tools (`web_search_tool_result`, `web_fetch_tool_result`, `code_execution_tool_result`, etc.) are dropped by the SDK's message parser. When this happens you may see `tool_call` parts for those tools without a matching `tool_call_response` — an upstream SDK limitation, not a gap in Logfire's instrumentation.

## Additional events captured

The integration also surfaces two non-message stream events as logs nested under the `invoke_agent` span:

- **Rate-limit transitions** (`RateLimitEvent`): emitted whenever the CLI reports a rate-limit state change (`allowed` → `allowed_warning` → `rejected`) or overage state, with `gen_ai.rate_limit.status`, `.type`, `.utilization`, `.resets_at`, and a `.raw` passthrough. `rejected` escalates the enclosing `invoke_agent` span to error level.
- **Session-store mirror errors** (`MirrorErrorMessage`): error-level logs when the configured `SessionStore` fails to mirror a transcript line. The local-disk transcript is unaffected.

## End-of-conversation result fields

The `invoke_agent` span carries the conversation-level summary extracted from `ResultMessage`:

- `duration_api_ms` — time spent waiting on the Anthropic API (vs `duration_ms`, the total wall clock).
- `gen_ai.response.finish_reasons` — the per-model stop reason as an OTel semconv array (e.g. `["end_turn"]`, `["max_tokens"]`).
- `claude.result.subtype` — conversation-level end state: `success` / `error_max_turns` / `error_during_execution`. Distinct from `finish_reasons`.
- `claude.result.text` — the final aggregated response text.
- `claude.result.errors` — list of diagnostic error strings encountered during the run, if any.
- `claude.result.structured_output` — the structured payload when `output_format` is configured.
- `claude.model_usage` — per-model token breakdown (useful when `fallback_model` kicks in and the aggregate `gen_ai.usage.*` no longer attributes tokens to a single model).
- `claude.permission_denials` — list of tool-permission denials recorded by the CLI.

The `claude.*` attributes except `claude.permission_denials` are added to Logfire's scrubber allowlist so model-generated content and user-supplied schemas pass through intact. `claude.permission_denials` entries contain the caller-supplied `tool_input` (arbitrary user data) and deliberately remain subject to default scrubbing.

## Agent Run view on the root span

The `invoke_agent` root span is finalised with the full conversation and aggregate metadata for dashboarding:

- `pydantic_ai.all_messages` — full conversation (user prompt, assistant turns, tool invocations, tool responses). This is the attribute that triggers Logfire's "Agent Run" chat-view rendering on the root span; without it the root only shows metadata and you have to drill into per-turn `chat` spans. The attribute name is a Logfire UI convention (originally established by `pydantic-ai`) and is already allowlisted by the scrubber.
- `claude.tools_used` — aggregated invocation counts as `[{"tool": name, "count": n}, ...]` across client, MCP, and server tools. Useful for dashboards that summarise tool usage per session without needing to traverse the message array.
- `gen_ai.agent.name` — the agent framework identifier (currently fixed to `claude-code`).
- `claude.cwd` — the working directory from `ClaudeAgentOptions.cwd`, when set. Intentionally under the `claude.*` namespace rather than `session.*` so value-level scrubbing (e.g. paths containing `secret` / `private_key` / `auth`) still applies.

### Session configuration

When the corresponding `ClaudeAgentOptions` field is set, the following attributes also appear on the `invoke_agent` root span. They're useful for filtering / grouping sessions in dashboards by configuration.

- `claude.options.model` / `claude.options.fallback_model` — the model originally requested (distinct from `gen_ai.request.model` on the `chat` span which reflects the *actually used* model).
- `claude.permission_mode`, `claude.max_turns`, `claude.max_budget_usd`, `claude.effort`.
- `claude.allowed_tools`, `claude.disallowed_tools`, `claude.setting_sources`, `claude.agents` (custom-agent names only — `AgentDefinition` bodies are deliberately not serialised).
- `claude.skills_mode` (`"all"` | `"allowlist"`) plus `claude.skills` (the list, in allowlist mode). The mixed `Literal["all"] | list[str]` SDK shape is normalised to a stable list type plus a discriminator so column-typed downstream stores see one schema.
- `claude.continue_conversation`, `claude.include_partial_messages`, `claude.enable_file_checkpointing` — booleans, emitted only when `True` to keep happy-path spans noise-free.
- `claude.resume_from` (from `ClaudeAgentOptions.resume`) and `claude.fork_on_resume` (from `fork_session`). Both renamed to drop the `session` substring that would otherwise trip the default scrubber's attribute-name match.
- `ClaudeAgentOptions.user` is intentionally **not** surfaced. Although the SDK forces it to be a real Unix username, it remains host-level identity that users may legitimately want to keep out of every span. Operators who want it can record it themselves outside this integration.

## Per-turn chat span attributes

Each `chat` child span carries per-turn identifiers derived from the `AssistantMessage` that closed the turn:

- `gen_ai.response.id` — the Anthropic API message id (e.g. `msg_01ABC...`). Directly queryable from the Anthropic API console.
- `gen_ai.response.finish_reasons` — the per-turn stop reason as a single-element array (`["end_turn"]`, `["tool_use"]`, `["max_tokens"]`, …). Also surfaced on the `invoke_agent` root span from `ResultMessage.stop_reason` at conversation level — filter by span name when aggregating to avoid double-counting.
- `claude.message.uuid` — the SDK-side stream UUID for the AssistantMessage. Distinct from `gen_ai.response.id` (the API id): useful for correlating with SDK-local logs or offline replays that don't hit the Anthropic API.
- `claude.parent_tool_use_id` — set when an `AssistantMessage` is produced by a subagent spawned via a ToolUse; carries the parent ToolUse id for stitching subagent turns back to the parent agent's tool call.

When the SDK emits multiple `AssistantMessage`s on the same chat span (text followed by a tool_use call from one API response, for example), these per-turn identifiers are last-write-wins — the final `message_id` / `stop_reason` is authoritative for the turn, matching the accumulating behaviour of `gen_ai.output.messages`.

The resulting trace looks like this in Logfire:

![Logfire Claude Agent SDK Trace](../../images/logfire-screenshot-claude-agent-sdk.png)
