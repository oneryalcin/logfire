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

The integration surfaces non-message stream events as logs nested under the `invoke_agent` span. Stream-source events:

- **Rate-limit transitions** (`RateLimitEvent`): emitted whenever the CLI reports a rate-limit state change (`allowed` → `allowed_warning` → `rejected`) or overage state, with `gen_ai.rate_limit.status`, `.type`, `.utilization`, `.resets_at`, and a `.raw` passthrough. `rejected` escalates the enclosing `invoke_agent` span to error level.
- **Session-store mirror errors** (`MirrorErrorMessage`): error-level logs when the configured `SessionStore` fails to mirror a transcript line. The local-disk transcript is unaffected.

Hook-callback events (registered automatically on the `ClaudeAgentOptions.hooks` mapping):

- **`Hook: UserPromptSubmit`** (info): emitted on every `client.query()` with `claude.user_prompt` carrying the submitted text and `gen_ai.conversation.id`. The prompt attribute deliberately stays subject to value-level scrubbing — user-typed prompts may contain credentials and default-on redaction is the safer privacy posture; deployments that need raw prompts can opt-out per-attribute via a `scrubbing_callback`.
- **`Hook: Stop`** (info): emitted at end-of-turn with `claude.stop.last_assistant_message` (the verbatim final assistant text — model-generated content, allowlisted from scrubbing) and `claude.stop.hook_active` (only when `True`). The `last_assistant_message` field is captured defensively via `.get()` because the SDK type definition doesn't declare it; the integration silently skips it if a future SDK release renames the wire field.
- **`Hook: PreCompact`** (warn): emitted before the CLI compacts the context window. Carries `claude.compact.trigger` (`"manual"` for `/compact` or `"auto"` for context-pressure triggered) and `claude.compact.custom_instructions` (operator-set guidance text, allowlisted from scrubbing). Warn level keeps it filterable / alertable as the strongest leading indicator of context pressure on long sessions.
- **`Hook: Notification`** (info): emitted for CLI-emitted notifications. Carries `claude.notification.message`, `claude.notification.title` (when set), and `claude.notification.type`. The message attribute stays subject to default scrubbing because CLI-emitted text can evolve; allowlisting it would risk masking future regressions.
- **`Hook: PermissionRequest`** (info): emitted as an audit log when the CLI requests permission to invoke a tool. Carries `gen_ai.tool.name`, `claude.permission_request.tool_input` (NOT allowlisted — caller-supplied arbitrary data, mirroring the `claude.permission_denials` precedent), `claude.permission_request.suggestions` when non-empty, and subagent attribution (`claude.agent_id` / `claude.agent_type`) when the request originates inside a subagent context.
- **`Hook: PermissionResult`** (info on `allow`, warn on `deny`): emitted after your `ClaudeAgentOptions.can_use_tool` callback returns, capturing the outcome paired with the request. Always carries `gen_ai.tool.name`, `gen_ai.tool.call.id`, and `claude.permission_result.behavior` (`"allow"` / `"deny"` / `"unknown"` for forward-compat). On allow, surfaces `claude.permission_result.updated_input` and `claude.permission_result.updated_permissions` only when the callback mutated either (happy-path Allow logs stay minimal). On deny, additionally carries `claude.permission_result.message` and `claude.permission_result.interrupt` (only when `True`), and the still-open `execute_tool` span — denied calls fire neither `PreToolUseFailure` nor `PostToolUse`, so the integration also marks that span with `error.type='PermissionDeny'`, escalates it to warn level, and ends it promptly so its `duration_ms` reflects permission-check latency rather than the rest of the conversation.

The `Hook:` prefix is a deliberate UX choice for trace readability — it makes hook-callback events visually distinct from SDK-generated stream events when scanning a session.

!!! note "When the audit-flow hooks fire"
    `UserPromptSubmit` and `Stop` fire on every query / turn and are always observable. `PermissionRequest` and `PermissionResult` fire whenever `ClaudeAgentOptions.can_use_tool` is set — independent of transport mode. `Notification` and `PreCompact` are gated on interactive-TTY presence in the CLI and typically don't fire during scripted / non-interactive `ClaudeSDKClient` sessions; the integration captures them when they do fire (e.g. inside an interactive terminal session) without requiring any opt-in.

## Tool-input mutations on `execute_tool` spans

`PreToolUse` hooks can return `updatedInput` to mutate a tool's arguments before execution; the same applies to `can_use_tool`'s `PermissionResultAllow.updated_input`. When this happens between `pre_tool_use_hook` and execution, the integration overwrites `gen_ai.tool.call.arguments` on the `execute_tool` span with the executed (post-mutation) value — matching the OTel Gen AI semconv expectation that `arguments` reflects what the tool actually saw — and surfaces the original (pre-hook) value under `claude.tool_call.arguments.original`. The diff attribute is absent on the common path where no hook mutated the input, so happy-path spans stay noise-free. Both attributes carry caller-supplied data and remain subject to the default scrubber.

## Lifecycle and control-method spans

`ClaudeSDKClient`'s lifecycle entry/exit and mid-session control methods are wrapped in spans so startup latency, model switches, and MCP toggles are visible — none of these were traced before:

- **`connect`** — spans the CLI subprocess launch and `Query.initialize` handshake. Carries `claude.cli_path` when set in `ClaudeAgentOptions`. On `CLINotFoundError` / `CLIConnectionError` / `ProcessError`, the span is escalated to error level with `error.type` (short class name) and, for `ProcessError` specifically, `claude.process.exit_code` and `claude.process.stderr` (truncated to the first 200 characters). Logfire's automatic exception recording also attaches the full traceback under `exception_type` / `exception.stacktrace` separately, so the audit trail is complete.
- **`disconnect`** — spans subprocess teardown. The SDK's `connect()` calls `disconnect()` from its own except cleanup on failure; that nested invocation produces a child `disconnect` span under the failed `connect`.
- **`set_model`** — captures the requested model under `gen_ai.request.model`. Mid-session model switches were previously invisible, breaking cost attribution when the active model differed from `ClaudeAgentOptions.model`. The attribute is absent when `set_model(None)` is called to revert to the default.
- **`set_permission_mode`** — captures the new mode under `claude.permission_mode` (the same attribute name used by the options-side capture from #8).
- **`rewind_files`** — captures `claude.rewind.user_message_id`.
- **`stop_task`** — captures `claude.task_id`.
- **`interrupt`** — timing only, no input attributes.
- **`mcp.reconnect`** / **`mcp.toggle`** — both carry `claude.mcp.server_name`; `mcp.toggle` additionally carries `claude.mcp.enabled`. The `mcp.` prefix groups MCP sub-domain operations distinctly from session-lifecycle calls in dashboards.

These spans are top-level (no `invoke_agent` parent) when called between turns; if a control method is invoked from inside an active `receive_response` / `receive_messages` iteration (e.g. from a hook callback), it nests under the active `invoke_agent` via the existing thread-local OTel context.

`receive_messages` (the alternate iterator to `receive_response`) now produces the same `invoke_agent` / `chat` / `Hook: *` span tree as `receive_response`. Sessions using `receive_messages` were previously a complete observability black hole; the integration internally factors a shared lifecycle so both iterators produce identical traces.

!!! note "`receive_messages` and the break-without-aclose edge case"
    `receive_messages` has no auto-stop sentinel, so the user must `break` to leave the loop. If the loop breaks without explicitly calling `await iter.aclose()` or using `async with closing(...)`, Python may not run the wrapper's `finally` block immediately — the OTel context for `invoke_agent` stays active until garbage collection runs, and any operations called between break and aclose (including `__aexit__`'s implicit `disconnect`) will parent to the stale `invoke_agent`. Doesn't affect `receive_response` (which auto-stops cleanly at `ResultMessage`). For `receive_messages`, prefer `async with contextlib.aclosing(client.receive_messages()) as it: ...` if you break early.

## Subagent observability

When the Claude CLI spawns a subagent via the Task tool, the integration opens a `subagent {agent_type}` child span under `invoke_agent` keyed by `agent_id` (`SubagentStart` hook). The span closes on the matching `SubagentStop`. Tool calls fired inside the subagent context re-parent their `execute_tool` child spans under the subagent span via `agent_id`, and per-turn `chat` spans emitted while the subagent is running re-parent under the subagent span via the `parent_tool_use_id` carried on subagent-originating `UserMessage` / `AssistantMessage` (issue #26). The trace tree looks like:

```
invoke_agent                      (root, parent thread)
├── chat claude-sonnet-4-6        (parent's turn that emits the Task tool call)
├── execute_tool Agent            (parent's view of the Task tool — current SDK
│                                  surfaces the Task tool as 'Agent')
├── subagent echo-helper          (sibling of execute_tool Agent)
│   ├── chat claude-haiku-…       (subagent's per-turn chat span, nested via #26)
│   ├── execute_tool Bash         (subagent's tool call, re-parented via agent_id)
│   └── chat claude-haiku-…       (subagent's continuation after the tool result)
├── Task started                  (log under invoke_agent, info)
├── Task progress                 (log; only fires for tool-using or background subagents)
├── Task completed                (log; level depends on status)
└── chat claude-sonnet-4-6        (parent's continuation)
```

Subagent attribution carried on the span:

- `gen_ai.agent.name`, `claude.agent_id` (SDK-generated, opaque), `claude.agent_type` — the disambiguators when multiple subagents run in parallel.
- `claude.subagent.last_assistant_message` — the verbatim final subagent text (allowlisted from scrubbing — model-generated content, mirrors `claude.stop.last_assistant_message`).
- `claude.agent.transcript_path` — the subagent's own transcript JSONL, distinct from the parent session's.
- When `ClaudeAgentOptions.agents[agent_type]` is configured (custom subagent definition), the full `AgentDefinition` metadata: `claude.agent.model`, `.description`, `.system_prompt` (from `AgentDefinition.prompt`, allowlisted), `.initial_prompt` (allowlisted), `.tools`, `.disallowed_tools`, `.skills`, `.memory`, `.background`, plus `claude.permission_mode` / `claude.max_turns` / `claude.effort` (re-using #8's constants — same configuration concepts; `span_name` distinguishes parent-options from subagent-definition values in dashboards).

Three system-message events emitted during the subagent run surface as logs under `invoke_agent`:

- **`Task started`** (info): carries `claude.task_id`, `claude.task.description`, `claude.task.type`, plus `gen_ai.tool.call.id` (the parent's Task tool id — for cross-span correlation).
- **`Task progress`** (info): carries `claude.task.usage` (a TaskUsage dict — `total_tokens` / `tool_uses` / `duration_ms`) and `claude.task.last_tool_name`. Typically only fires for long-running / tool-using subagents; pure-text foreground subagents go straight from started to notification.
- **`Task {status}`** (level depends on status): `completed` → info, `stopped` → warn, `failed` → error. Carries `claude.task.summary` (allowlisted — model-generated end-of-task text), `claude.task.output_file`, optional `claude.task.usage`. Does NOT escalate the parent `invoke_agent` span on `failed` — the parent isn't logically failed, just the subagent.

`claude.subagent_count` lands on the root `invoke_agent` at conversation close (mirrors `claude.tools_used`), giving dashboards a per-session subagent-spawn count.

### Parallel subagents

Multiple Task dispatches in a single parent turn produce multiple concurrent `subagent` spans, each keyed by its own `agent_id`. The integration's per-conversation `subagent_spans` dict disambiguates by `agent_id`, so each subagent's tool calls correctly nest under its own span even when their hook callbacks interleave over the SDK's control channel. Chat spans emitted by parallel subagents disambiguate the same way: each subagent-originating `UserMessage` carries a unique `parent_tool_use_id` that resolves through the `parent_tool_use_id_to_agent_id` map (populated by `TaskStartedMessage`) to the right subagent span.

Re-parented chat spans also carry `claude.in_subagent.agent_id` and `claude.in_subagent.agent_type` for queryability — useful in renderers that flatten the trace tree, and for SQL filters like `WHERE attributes->>'claude.in_subagent.agent_type' = 'reviewer'`.

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
- `claude.user_prompt_count` / `claude.compact_count` / `claude.notification_count` / `claude.permission_request_count` — per-conversation cumulative counts of the respective hook-callback events. Emitted only when non-zero so happy-path spans don't carry always-zero attrs. The bare `_count` suffix (rather than nesting under each event's sub-namespace) is intentional — these are root-span session aggregates, mirroring the `claude.tools_used` precedent.
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
