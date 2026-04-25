from __future__ import annotations

import functools
import threading
from collections import Counter
from collections.abc import AsyncGenerator, AsyncIterable
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import TYPE_CHECKING, Any, cast

import claude_agent_sdk
from claude_agent_sdk import (
    AssistantMessage,
    HookMatcher,
    ResultMessage,
    ServerToolResultBlock,
    ServerToolUseBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from claude_agent_sdk.types import HookContext, SyncHookJSONOutput

# Optional message types — resolved at module load from the installed SDK so
# older versions that lack them still import cleanly. Using getattr avoids
# a redundant try/except on a module we've already imported above.
RateLimitEvent = getattr(claude_agent_sdk, 'RateLimitEvent', None)
MirrorErrorMessage = getattr(claude_agent_sdk, 'MirrorErrorMessage', None)

from opentelemetry import context as context_api, trace as trace_api

from logfire._internal.integrations.llm_providers.semconv import (
    AGENT_NAME,
    CLAUDE_AGENT_ID,
    CLAUDE_AGENT_TYPE,
    CLAUDE_AGENTS,
    CLAUDE_ALLOWED_TOOLS,
    CLAUDE_COMPACT_COUNT,
    CLAUDE_COMPACT_INSTRUCTIONS,
    CLAUDE_COMPACT_TRIGGER,
    CLAUDE_CONTINUE_CONVERSATION,
    CLAUDE_CWD,
    CLAUDE_DISALLOWED_TOOLS,
    CLAUDE_EFFORT,
    CLAUDE_ENABLE_FILE_CHECKPOINTING,
    CLAUDE_FORK_ON_RESUME,
    CLAUDE_INCLUDE_PARTIAL_MESSAGES,
    CLAUDE_MAX_BUDGET_USD,
    CLAUDE_MAX_TURNS,
    CLAUDE_MESSAGE_UUID,
    CLAUDE_MODEL_USAGE,
    CLAUDE_NOTIFICATION_COUNT,
    CLAUDE_NOTIFICATION_MESSAGE,
    CLAUDE_NOTIFICATION_TITLE,
    CLAUDE_NOTIFICATION_TYPE,
    CLAUDE_OPTIONS_FALLBACK_MODEL,
    CLAUDE_OPTIONS_MODEL,
    CLAUDE_PARENT_TOOL_USE_ID,
    CLAUDE_PERMISSION_DENIALS,
    CLAUDE_PERMISSION_MODE,
    CLAUDE_PERMISSION_REQUEST_COUNT,
    CLAUDE_PERMISSION_REQUEST_SUGGESTIONS,
    CLAUDE_PERMISSION_REQUEST_TOOL_INPUT,
    CLAUDE_PERMISSION_RESULT_BEHAVIOR,
    CLAUDE_PERMISSION_RESULT_INTERRUPT,
    CLAUDE_PERMISSION_RESULT_MESSAGE,
    CLAUDE_PERMISSION_RESULT_UPDATED_INPUT,
    CLAUDE_PERMISSION_RESULT_UPDATED_PERMISSIONS,
    CLAUDE_RESULT_ERRORS,
    CLAUDE_RESULT_STRUCTURED_OUTPUT,
    CLAUDE_RESULT_SUBTYPE,
    CLAUDE_RESULT_TEXT,
    CLAUDE_RESUME_FROM,
    CLAUDE_SETTING_SOURCES,
    CLAUDE_SKILLS,
    CLAUDE_SKILLS_MODE,
    CLAUDE_STOP_HOOK_ACTIVE,
    CLAUDE_STOP_LAST_ASSISTANT_MESSAGE,
    CLAUDE_TOOL_CALL_ARGUMENTS_ORIGINAL,
    CLAUDE_TOOLS_USED,
    CLAUDE_USER_PROMPT,
    CLAUDE_USER_PROMPT_COUNT,
    CONVERSATION_ID,
    ERROR_TYPE,
    INPUT_MESSAGES,
    OPERATION_NAME,
    OUTPUT_MESSAGES,
    PROVIDER_NAME,
    PYDANTIC_AI_ALL_MESSAGES,
    RATE_LIMIT_OVERAGE_DISABLED_REASON,
    RATE_LIMIT_OVERAGE_RESETS_AT,
    RATE_LIMIT_OVERAGE_STATUS,
    RATE_LIMIT_RAW,
    RATE_LIMIT_RESETS_AT,
    RATE_LIMIT_STATUS,
    RATE_LIMIT_TYPE,
    RATE_LIMIT_UTILIZATION,
    REQUEST_MODEL,
    RESPONSE_FINISH_REASONS,
    RESPONSE_ID,
    RESPONSE_MODEL,
    SYSTEM,
    SYSTEM_INSTRUCTIONS,
    TOOL_CALL_ARGUMENTS,
    TOOL_CALL_ID,
    TOOL_CALL_RESULT,
    TOOL_NAME,
    ChatMessage,
    MessagePart,
    OutputMessage,
    ReasoningPart,
    TextPart,
    ToolCallPart,
    ToolCallResponsePart,
)
from logfire._internal.utils import handle_internal_errors

if TYPE_CHECKING:
    # String forward refs for the SDK types that may be absent at runtime on
    # older SDK versions (see getattr above).
    from claude_agent_sdk import MirrorErrorMessage as _MirrorErrorMessage
    from claude_agent_sdk import RateLimitEvent as _RateLimitEvent

    from logfire._internal.main import Logfire, LogfireSpan


# ---------------------------------------------------------------------------
# Thread-local storage for per-conversation state.
#
# The Claude Agent SDK uses anyio internally, and anyio tasks don't propagate
# contextvars from the parent. This means OTel's context propagation breaks
# for hook callbacks. We use threading.local() as a workaround — storing a
# single _ConversationState object that hooks retrieve.
# ---------------------------------------------------------------------------
_thread_local = threading.local()


def _get_state() -> _ConversationState | None:
    return getattr(_thread_local, 'state', None)


def _set_state(state: _ConversationState) -> None:
    _thread_local.state = state


def _clear_state() -> None:
    if hasattr(_thread_local, 'state'):  # pragma: no branch
        delattr(_thread_local, 'state')


# ---------------------------------------------------------------------------
# Utility functions for converting SDK types to semconv part dicts.
# ---------------------------------------------------------------------------


def _content_blocks_to_output_messages(content: Any) -> list[OutputMessage]:
    """Convert SDK content block objects into semconv OutputMessages."""
    parts: list[MessagePart] = []
    if not isinstance(content, list):
        return []

    for block in cast('list[Any]', content):
        if isinstance(block, TextBlock):
            parts.append(TextPart(type='text', content=block.text))
        elif isinstance(block, ThinkingBlock):
            parts.append(ReasoningPart(type='reasoning', content=block.thinking))
        elif isinstance(block, (ToolUseBlock, ServerToolUseBlock)):
            # ServerToolUseBlock mirrors ToolUseBlock's shape; `name` discriminates
            # server tools (web_search, code_execution, etc.). Note: the SDK's
            # message parser currently only constructs ServerToolResultBlock for
            # `advisor_tool_result`; results for other server tools are dropped
            # upstream, so those `tool_call` parts may appear without a matching
            # `tool_call_response` — an SDK limitation, not an integration bug.
            part = ToolCallPart(
                type='tool_call',
                id=block.id or '',
                name=block.name or '',
            )
            part['arguments'] = block.input
            parts.append(part)
        elif isinstance(block, (ToolResultBlock, ServerToolResultBlock)):
            parts.append(
                ToolCallResponsePart(
                    type='tool_call_response',
                    id=block.tool_use_id or '',
                    response=block.content,  # type: ignore
                )
            )
        else:
            parts.append(block)

    msg = OutputMessage(role='assistant', parts=parts)
    return [msg]


def _extract_usage(usage: Any, *, partial: bool = False) -> dict[str, int]:
    """Extract usage metrics from a Claude usage object or dict.

    Args:
        usage: A usage object or dict from the SDK.
        partial: If True, prefix attribute names with ``gen_ai.usage.partial.``
            instead of ``gen_ai.usage.``. Used for chat spans where per-message
            usage from the SDK is unreliable.
    """
    if not usage:
        return {}

    def get(key: str) -> Any:
        if isinstance(usage, dict):
            return cast(dict[str, Any], usage).get(key)
        return getattr(usage, key, None)

    def to_int(value: Any) -> int | None:
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    prefix = 'gen_ai.usage.partial.' if partial else 'gen_ai.usage.'

    result: dict[str, int] = {}

    # input_tokens is the *total* input token count.
    # The Anthropic API's input_tokens only counts uncached tokens,
    # so we sum input + cache_read + cache_creation to get the actual total.
    input_tokens = to_int(get('input_tokens')) or 0
    cache_read = to_int(get('cache_read_input_tokens')) or 0
    cache_creation = to_int(get('cache_creation_input_tokens')) or 0
    total_input = input_tokens + cache_read + cache_creation
    if total_input:
        result[f'{prefix}input_tokens'] = total_input

    if (v := to_int(get('output_tokens'))) is not None:
        result[f'{prefix}output_tokens'] = v

    if cache_read:
        result[f'{prefix}cache_read.input_tokens'] = cache_read
    if cache_creation:
        result[f'{prefix}cache_creation.input_tokens'] = cache_creation

    return result


def _options_attrs(options: Any) -> dict[str, Any]:
    """Extract observability-relevant fields from ``ClaudeAgentOptions``.

    Only fields that help dashboards / audits / debugging are surfaced.
    Omits infrastructure-only fields (``cli_path``, ``env``, ``stderr``,
    ``hooks``, ``session_store``, ``plugins``, ``sandbox``,
    ``load_timeout_ms``), callables, and potentially-large configs
    (``mcp_servers``, ``extra_args``, ``output_format`` schemas,
    ``thinking`` / ``task_budget`` dicts).

    Also intentionally omits ``user``: although the SDK calls ``getpwnam``
    on it (forcing a real Unix username), it's host-level identity that
    operators may legitimately want to keep out of every span. Operators
    who want it can capture ``user`` themselves outside this integration.

    ``resume`` and ``fork_session`` are surfaced under renamed keys
    (``claude.resume_from`` / ``claude.fork_on_resume``) because the
    original names contain the ``session`` substring which would trigger
    the default logfire scrubber on the attribute name.
    """
    # Defensive isinstance — duck-typed mocks could otherwise silently
    # emit garbage as attribute values (e.g. ``model=123`` ints).
    if not isinstance(options, claude_agent_sdk.ClaudeAgentOptions):
        return {}

    attrs: dict[str, Any] = {}

    # Scalars — emit when not None.
    scalar_map: tuple[tuple[str, str], ...] = (
        ('model', CLAUDE_OPTIONS_MODEL),
        ('fallback_model', CLAUDE_OPTIONS_FALLBACK_MODEL),
        ('permission_mode', CLAUDE_PERMISSION_MODE),
        ('max_turns', CLAUDE_MAX_TURNS),
        ('max_budget_usd', CLAUDE_MAX_BUDGET_USD),
        ('effort', CLAUDE_EFFORT),
        ('resume', CLAUDE_RESUME_FROM),
    )
    for src, dst in scalar_map:
        if (value := getattr(options, src, None)) is not None:
            attrs[dst] = value

    # ``skills`` is ``list[str] | Literal["all"] | None``. Normalise the
    # mixed-shape into a stable list-typed ``claude.skills`` plus a
    # discriminator under ``claude.skills_mode`` so downstream typed
    # stores (column-typed warehouses) see one shape.
    if (skills := getattr(options, 'skills', None)) is not None:
        if skills == 'all':
            attrs[CLAUDE_SKILLS_MODE] = 'all'
        else:
            attrs[CLAUDE_SKILLS_MODE] = 'allowlist'
            attrs[CLAUDE_SKILLS] = list(skills)

    # Lists — emit when non-empty.
    list_map: tuple[tuple[str, str], ...] = (
        ('allowed_tools', CLAUDE_ALLOWED_TOOLS),
        ('disallowed_tools', CLAUDE_DISALLOWED_TOOLS),
        ('setting_sources', CLAUDE_SETTING_SOURCES),
    )
    for src, dst in list_map:
        if value := getattr(options, src, None):
            attrs[dst] = list(value)

    # Booleans — emit only when True (non-default).
    bool_map: tuple[tuple[str, str], ...] = (
        ('continue_conversation', CLAUDE_CONTINUE_CONVERSATION),
        ('include_partial_messages', CLAUDE_INCLUDE_PARTIAL_MESSAGES),
        ('enable_file_checkpointing', CLAUDE_ENABLE_FILE_CHECKPOINTING),
        ('fork_session', CLAUDE_FORK_ON_RESUME),
    )
    for src, dst in bool_map:
        if getattr(options, src, False):
            attrs[dst] = True

    # ``agents`` is a dict of ``name → AgentDefinition``. Emit names only;
    # only handle the dict shape because future SDK changes (e.g. to a
    # ``list[AgentDefinition]``) would otherwise silently leak object
    # reprs into the attribute.
    agents = getattr(options, 'agents', None)
    if isinstance(agents, dict) and agents:
        attrs[CLAUDE_AGENTS] = sorted(agents.keys())

    return attrs


# ---------------------------------------------------------------------------
# Hook callbacks for tool call tracing.
# ---------------------------------------------------------------------------


async def pre_tool_use_hook(
    input_data: Any,
    tool_use_id: str | None,
    _context: HookContext,
) -> SyncHookJSONOutput:
    """Create a child span when a tool execution starts."""
    # Truthy-check covers ``None`` *and* empty string. The SDK declares
    # ``tool_use_id: str | None`` and an empty string would silently skip
    # both the span open and the issue-#10 diff snapshot — acceptable
    # because there's no way to correlate a tool call without an id anyway.
    if not tool_use_id:
        return {}

    with handle_internal_errors:
        state = _get_state()
        if state is None:
            return {}

        tool_name = str(input_data.get('tool_name', 'unknown_tool'))
        tool_input = input_data.get('tool_input', {})
        # Close the current chat span so it doesn't overlap with tool execution.
        state.close_chat_span()

        # Temporarily attach root span context so the new span is parented correctly,
        # then immediately detach. We can't keep the context attached because hooks
        # run in different async contexts (anyio tasks) and detaching later would fail.
        otel_span = state.root_span._span  # pyright: ignore[reportPrivateUsage]
        if otel_span is None:  # pragma: no cover
            return {}
        parent_ctx = trace_api.set_span_in_context(otel_span)
        token = context_api.attach(parent_ctx)
        try:
            span_name = f'execute_tool {tool_name}'
            span = state.logfire.span(span_name)
            span.set_attributes(
                {
                    OPERATION_NAME: 'execute_tool',
                    TOOL_NAME: tool_name,
                    TOOL_CALL_ID: tool_use_id,
                    TOOL_CALL_ARGUMENTS: tool_input,
                }
            )
            span._start()  # pyright: ignore[reportPrivateUsage]
            state.active_tool_spans[tool_use_id] = span
            # Snapshot for issue #10's pre-vs-executed input diff. Only kept
            # when the input is dict-shaped — non-dict shapes (rare/malformed)
            # would compare unreliably and aren't worth the diff.
            # Shallow ``dict(...)`` copy is sufficient because the SDK
            # round-trips ``tool_input`` through JSON at the control-protocol
            # boundary, so nested dicts are fresh on each callback. If a
            # future SDK refactor bypasses JSON (e.g. for in-process MCP),
            # nested-dict mutations would silently miss the diff — revisit
            # with ``copy.deepcopy`` then.
            if isinstance(tool_input, dict):
                state.original_tool_inputs[tool_use_id] = dict(tool_input)
        finally:
            context_api.detach(token)

    return {}


async def post_tool_use_hook(
    input_data: Any,
    tool_use_id: str | None,
    _context: HookContext,
) -> SyncHookJSONOutput:
    """End the tool span after successful execution."""
    if not tool_use_id:
        return {}

    with handle_internal_errors:
        state = _get_state()
        if state is None:
            return {}

        span = state.active_tool_spans.pop(tool_use_id, None)
        # Consume the issue-#10 pre-hook input snapshot (None if the
        # snapshot path skipped — see ``pre_tool_use_hook`` for the
        # shallow-copy / dict-shape gate).
        original_input = state.original_tool_inputs.pop(tool_use_id, None)
        if not span:  # pragma: no cover
            return {}

        # Issue #10: a hook (PreToolUse output ``updatedInput`` or
        # ``can_use_tool``'s ``PermissionResultAllow.updated_input``) may have
        # mutated the input between pre and execute. Overwrite the OTel-
        # canonical attribute with the executed value and surface the
        # original under ``claude.tool_call.arguments.original``.
        executed_input = input_data.get('tool_input')
        if (
            isinstance(executed_input, dict)
            and isinstance(original_input, dict)
            and executed_input != original_input
        ):
            span.set_attribute(TOOL_CALL_ARGUMENTS, executed_input)
            span.set_attribute(CLAUDE_TOOL_CALL_ARGUMENTS_ORIGINAL, original_input)

        tool_response = input_data.get('tool_response')
        if tool_response is not None:
            span.set_attribute(TOOL_CALL_RESULT, tool_response)
        span._end()  # pyright: ignore[reportPrivateUsage]

        # Record tool result for the next chat span's input messages
        tool_name = str(input_data.get('tool_name', 'unknown_tool'))
        state.add_tool_result(tool_use_id, tool_name, tool_response if tool_response is not None else '')

    return {}


async def post_tool_use_failure_hook(
    input_data: Any,
    tool_use_id: str | None,
    _context: HookContext,
) -> SyncHookJSONOutput:
    """End the tool span with an error after failed execution."""
    if not tool_use_id:
        return {}

    with handle_internal_errors:
        state = _get_state()
        if state is None:  # pragma: no cover
            return {}

        span = state.active_tool_spans.pop(tool_use_id, None)
        # Discard any pre-hook input snapshot kept by issue #10's diff path
        # (no executed input on failure, no diff to surface).
        state.original_tool_inputs.pop(tool_use_id, None)
        if not span:  # pragma: no cover
            return {}

        error = str(input_data.get('error', 'Unknown error'))
        span.set_attribute(ERROR_TYPE, error)
        span.set_level('error')
        span._end()  # pyright: ignore[reportPrivateUsage]

        # Record the error as a tool result so the next turn's input is complete.
        tool_name = str(input_data.get('tool_name', 'unknown_tool'))
        state.add_tool_result(tool_use_id, tool_name, error)

    return {}


# ---------------------------------------------------------------------------
# Hook callbacks for non-tool-lifecycle events (issue #9).
#
# Empirically the SDK passes a freshly-allocated UUID as ``tool_use_id`` for
# these events too — so the ``if not tool_use_id`` early-bail used by the
# tool-lifecycle hooks above is intentionally absent here. Each callback is a
# thin shim that runs a matching ``_record_*`` helper inside the root span's
# OTel context (hooks run in different anyio tasks → contextvars don't
# propagate; without explicit attach the emitted log lands as an orphan
# top-level span rather than nested under ``invoke_agent``).
# ---------------------------------------------------------------------------


def _attach_root_context(state: _ConversationState) -> Any:
    """Temporarily attach the root span as the active OTel context.

    Mirrors the attach/detach pattern used by ``pre_tool_use_hook``. Returns
    a token to pass to ``context_api.detach`` once the emission is done.
    Returns ``None`` if the span is missing — caller should noop.
    """
    otel_span = state.root_span._span  # pyright: ignore[reportPrivateUsage]
    if otel_span is None:  # pragma: no cover
        return None
    parent_ctx = trace_api.set_span_in_context(otel_span)
    return context_api.attach(parent_ctx)


async def _run_hook(
    record_fn: Any,
    input_data: Any,
) -> None:
    """Shared boilerplate for the 5 issue-#9 hook callbacks.

    Each callback runs ``record_fn(state, input_data)`` inside the root
    span's OTel context (hooks run in different anyio tasks so contextvars
    don't propagate; without explicit attach the emitted log lands as an
    orphan top-level span rather than nested under ``invoke_agent``).

    Issue #10's ``_wrap_can_use_tool`` deliberately does NOT use this
    helper — its shape is ``await user_callback() -> emit -> return result``
    (await-then-emit, with the user-supplied result threaded back to the
    SDK), whereas this helper is pure emission.
    """
    with handle_internal_errors:
        state = _get_state()
        if state is None:  # pragma: no cover
            return
        token = _attach_root_context(state)
        if token is None:  # pragma: no cover
            return
        try:
            record_fn(state, input_data)
        finally:
            context_api.detach(token)


async def user_prompt_submit_hook(
    input_data: Any,
    _tool_use_id: str | None,
    _context: HookContext,
) -> SyncHookJSONOutput:
    """Surface the prompt the user just submitted as an info-level log."""
    await _run_hook(_record_user_prompt_submit, input_data)
    return {}


async def stop_hook(
    input_data: Any,
    _tool_use_id: str | None,
    _context: HookContext,
) -> SyncHookJSONOutput:
    """Surface the end-of-turn Stop event as an info-level log.

    Does **not** close the chat span here: ``_ConversationState`` already
    closes it on ``UserMessage`` / ``ResultMessage`` boundaries, and racing
    the close from a hook running in a different anyio task only adds risk
    for marginal value.
    """
    await _run_hook(_record_stop, input_data)
    return {}


async def pre_compact_hook(
    input_data: Any,
    _tool_use_id: str | None,
    _context: HookContext,
) -> SyncHookJSONOutput:
    """Surface upcoming context-window compaction as a warn-level log.

    Compaction is the strongest leading signal of context pressure on long
    sessions; warn level keeps it filterable in dashboards without being
    treated as an error.
    """
    await _run_hook(_record_pre_compact, input_data)
    return {}


async def notification_hook(
    input_data: Any,
    _tool_use_id: str | None,
    _context: HookContext,
) -> SyncHookJSONOutput:
    """Surface CLI-emitted notifications as info-level logs."""
    await _run_hook(_record_notification, input_data)
    return {}


async def permission_request_hook(
    input_data: Any,
    _tool_use_id: str | None,
    _context: HookContext,
) -> SyncHookJSONOutput:
    """Surface tool-permission *requests* (the ask, not the outcome) as info-level logs.

    The corresponding *outcome* path — ``PermissionResultDeny`` from the
    ``can_use_tool`` callback and ``PreToolUse.updatedInput`` mutations —
    is the territory of issue #10, not this hook. See the recon comment
    on issue #9 for the boundary rationale.
    """
    await _run_hook(_record_permission_request, input_data)
    return {}


# ---------------------------------------------------------------------------
# Instrumentation entry point.
# ---------------------------------------------------------------------------


def instrument_claude_agent_sdk(logfire_instance: Logfire) -> AbstractContextManager[None]:
    """Instrument the Claude Agent SDK by monkey-patching ClaudeSDKClient.

    Returns:
        A context manager that will revert the instrumentation when exited.
            This context manager doesn't take into account threads or other concurrency.
            Calling this function will immediately apply the instrumentation
            without waiting for the context manager to be opened,
            i.e. it's not necessary to use this as a context manager.
    """
    cls = claude_agent_sdk.ClaudeSDKClient

    if getattr(cls, '_is_instrumented_by_logfire', False):
        return nullcontext()

    original_init = cls.__init__
    original_query = cls.query
    original_receive_response = cls.receive_response

    cls._is_instrumented_by_logfire = True  # pyright: ignore[reportAttributeAccessIssue]

    logfire_claude = logfire_instance.with_settings(custom_scope_suffix='claude_agent_sdk')

    # --- Patch __init__ ---
    @functools.wraps(original_init)
    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)

        self._logfire_prompt = None

        if self.options:  # pragma: no branch
            _inject_tracing_hooks(self.options)

    cls.__init__ = patched_init

    # --- Patch query ---
    @functools.wraps(original_query)
    async def patched_query(self: Any, *args: Any, **kwargs: Any) -> Any:
        self._logfire_prompt = None
        prompt = args[0] if args else kwargs.get('prompt')

        if isinstance(prompt, str):
            self._logfire_prompt = prompt
        elif prompt is not None and not isinstance(prompt, AsyncIterable):  # pragma: no cover
            self._logfire_prompt = str(prompt)

        return await original_query(self, *args, **kwargs)

    cls.query = patched_query

    # --- Patch receive_response ---
    @functools.wraps(original_receive_response)
    async def patched_receive_response(self: Any) -> AsyncGenerator[Any, None]:
        prompt = getattr(self, '_logfire_prompt', None)
        input_messages: list[ChatMessage] = []
        if prompt:  # pragma: no branch
            input_messages = [ChatMessage(role='user', parts=[TextPart(type='text', content=prompt)])]

        span_data: dict[str, Any] = {
            OPERATION_NAME: 'invoke_agent',
            PROVIDER_NAME: 'anthropic',
            SYSTEM: 'anthropic',
            AGENT_NAME: 'claude-code',
        }
        if input_messages:  # pragma: no branch
            span_data[INPUT_MESSAGES] = input_messages
        if hasattr(self, 'options') and self.options:  # pragma: no branch
            system_prompt = getattr(self.options, 'system_prompt', None)
            if system_prompt:  # pragma: no branch
                text = str(system_prompt)
                span_data[SYSTEM_INSTRUCTIONS] = [TextPart(type='text', content=text)]
            if cwd := getattr(self.options, 'cwd', None):
                span_data[CLAUDE_CWD] = str(cwd)
            span_data.update(_options_attrs(self.options))

        with logfire_claude.span('invoke_agent', **span_data) as root_span:
            state = _ConversationState(
                logfire=logfire_claude,
                root_span=root_span,
                input_messages=input_messages,
                system_instructions=span_data.get(SYSTEM_INSTRUCTIONS),
            )
            _set_state(state)
            # Open the first chat span now — the LLM call starts at query time.
            state.open_chat_span()

            try:
                async for msg in original_receive_response(self):
                    with handle_internal_errors:
                        if isinstance(msg, AssistantMessage):
                            state.handle_assistant_message(msg)
                        elif isinstance(msg, UserMessage):
                            state.handle_user_message()
                        elif isinstance(msg, ResultMessage):
                            _record_result(root_span, msg)
                            if state.model:  # pragma: no branch
                                root_span.set_attribute(REQUEST_MODEL, state.model)
                                root_span.set_attribute(RESPONSE_MODEL, state.model)
                        elif RateLimitEvent is not None and isinstance(msg, RateLimitEvent):
                            _record_rate_limit_event(logfire_claude, root_span, msg)
                        elif MirrorErrorMessage is not None and isinstance(msg, MirrorErrorMessage):
                            _record_mirror_error(logfire_claude, msg)

                    yield msg
            finally:
                state.close()
                _clear_state()

    cls.receive_response = patched_receive_response

    @contextmanager
    def uninstrument_context():
        try:
            yield
        finally:
            cls.__init__ = original_init
            cls.query = original_query
            cls.receive_response = original_receive_response
            cls._is_instrumented_by_logfire = False  # pyright: ignore[reportAttributeAccessIssue]

    return uninstrument_context()


def _inject_tracing_hooks(options: Any) -> None:
    """Inject logfire tracing hooks into ClaudeAgentOptions and wrap any
    ``can_use_tool`` callback for outcome tracing (issue #10).

    Guards against duplicate injection / double-wrap when the same options
    object is reused across multiple ClaudeSDKClient instances.
    """
    if not hasattr(options, 'hooks'):
        return

    hooks: dict[str, list[HookMatcher]]
    if options.hooks is None:
        hooks = options.hooks = {}
    else:
        hooks = options.hooks
    with handle_internal_errors:
        # Guard against duplicate injection when the same options object is reused.
        if getattr(options, '_logfire_hooks_injected', False):  # pragma: no cover
            return
        options._logfire_hooks_injected = True

        # Tool-lifecycle hooks (existing) + non-tool-lifecycle hooks (issue #9).
        # The non-tool-lifecycle callbacks emit logfire log records under the
        # active ``invoke_agent`` span; they don't open child spans.
        events_to_callbacks: tuple[tuple[str, Any], ...] = (
            ('PreToolUse', pre_tool_use_hook),
            ('PostToolUse', post_tool_use_hook),
            ('PostToolUseFailure', post_tool_use_failure_hook),
            ('UserPromptSubmit', user_prompt_submit_hook),
            ('Stop', stop_hook),
            ('PreCompact', pre_compact_hook),
            ('Notification', notification_hook),
            ('PermissionRequest', permission_request_hook),
        )
        for event, callback in events_to_callbacks:
            hooks.setdefault(event, [])
            hooks[event].insert(0, HookMatcher(matcher=None, hooks=[callback]))

        # Wrap can_use_tool (issue #10). The user's callback is opt-in
        # (None for most users); only wrap when set so we don't inject a
        # callback the SDK would otherwise treat as a feature toggle.
        callback = getattr(options, 'can_use_tool', None)
        if callback is not None:
            options.can_use_tool = _wrap_can_use_tool(callback)


class _ConversationState:
    """Per-conversation state stored in thread-local during a receive_response iteration.

    Holds everything hooks need: the root span, logfire instance, active tool spans,
    chat span lifecycle, and conversation history. This keeps all mutable state in one
    object instead of scattered across globals and thread-local attributes.
    """

    def __init__(
        self,
        *,
        logfire: Logfire,
        root_span: LogfireSpan,
        input_messages: list[ChatMessage],
        system_instructions: list[TextPart] | None = None,
    ) -> None:
        self.logfire = logfire
        self.root_span = root_span
        self.active_tool_spans: dict[str, LogfireSpan] = {}
        # Tool input as seen by ``pre_tool_use_hook`` keyed by ``tool_use_id``.
        # ``post_tool_use_hook`` consumes it to detect updatedInput mutations
        # (issue #10). Cleared on consumption to bound size.
        self.original_tool_inputs: dict[str, Any] = {}
        self._current_span: LogfireSpan | None = None
        # Running conversation history — each chat span gets the full history as input.
        self._history: list[ChatMessage] = list(input_messages)
        # Track current span's output parts for merging consecutive messages.
        self._current_output_parts: list[MessagePart] = []
        self._system_instructions = system_instructions
        self.model: str | None = None
        # Hook-event counters (issue #9). Surfaced on the root span at
        # ``close()`` so dashboards can group by session-level activity.
        self.user_prompt_count = 0
        self.compact_count = 0
        self.notification_count = 0
        self.permission_request_count = 0

    def add_tool_result(self, tool_use_id: str, tool_name: str, result: Any) -> None:
        """Record a tool result to include in the next chat span's input messages."""
        msg = ChatMessage(
            role='tool',
            parts=[ToolCallResponsePart(type='tool_call_response', id=tool_use_id, name=tool_name, response=result)],
        )
        self._history.append(msg)

    def open_chat_span(self) -> None:
        """Open a new chat span — call when the LLM starts processing."""
        self.close_chat_span()

        span_data: dict[str, Any] = {
            OPERATION_NAME: 'chat',
            PROVIDER_NAME: 'anthropic',
            SYSTEM: 'anthropic',
        }
        if self._history:  # pragma: no branch
            span_data[INPUT_MESSAGES] = list(self._history)
        if self._system_instructions:  # pragma: no branch
            span_data[SYSTEM_INSTRUCTIONS] = self._system_instructions

        self._current_span = self.logfire.span('chat', **span_data)
        # Start without entering context — chat spans don't need to be on the
        # context stack, and this allows close_chat_span() to be called safely
        # from hooks running in different async contexts.
        self._current_span._start()  # pyright: ignore[reportPrivateUsage]
        self._current_output_parts = []

    def close_chat_span(self) -> None:
        """Close the current chat span without opening a new one.

        Safe to call from hooks (different async contexts) because chat spans
        are never entered into the OTel context stack.
        """
        if self._current_span is not None:
            if self._current_output_parts:  # pragma: no branch
                self._history.append(ChatMessage(role='assistant', parts=list(self._current_output_parts)))
            self._current_span._end()  # pyright: ignore[reportPrivateUsage]
            self._current_span = None
            self._current_output_parts = []

    def handle_user_message(self) -> None:
        """Handle UserMessage: open a new chat span for the next LLM call."""
        self.open_chat_span()

    def handle_assistant_message(self, message: AssistantMessage) -> None:
        """Handle AssistantMessage: add output and usage to the current chat span."""
        if self._current_span is None:  # pragma: no cover
            return

        content = getattr(message, 'content', [])
        output_messages = _content_blocks_to_output_messages(content)
        new_parts = output_messages[0]['parts'] if output_messages else []

        self._current_output_parts.extend(new_parts)
        self._current_span.set_attribute(
            OUTPUT_MESSAGES, [OutputMessage(role='assistant', parts=self._current_output_parts)]
        )

        model = getattr(message, 'model', None)
        if model:  # pragma: no branch
            self.model = model
            self._current_span.set_attribute(REQUEST_MODEL, model)
            self._current_span.set_attribute(RESPONSE_MODEL, model)
            # Update span name to include model.
            self._current_span.message = f'chat {model}'
            self._current_span.update_name(f'chat {model}')  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]

        usage = getattr(message, 'usage', None)
        if usage:  # pragma: no branch
            self._current_span.set_attributes(_extract_usage(usage, partial=True))

        # Per-turn identifiers; see the CLAUDE_* constants' comment for the
        # message_id / uuid / parent_tool_use_id distinction. When the SDK
        # emits multiple AssistantMessages on the same chat span (e.g. text
        # + tool_use from one API call), these are last-write-wins — the
        # final message_id/stop_reason is the authoritative one for the
        # turn, matching OUTPUT_MESSAGES which accumulates.
        attrs: dict[str, Any] = {}
        for src, dst in (
            ('message_id', RESPONSE_ID),
            ('uuid', CLAUDE_MESSAGE_UUID),
            ('parent_tool_use_id', CLAUDE_PARENT_TOOL_USE_ID),
        ):
            if (value := getattr(message, src, None)) is not None:
                attrs[dst] = value
        # OTel semconv: finish_reasons is an array even with a single value.
        if (stop_reason := getattr(message, 'stop_reason', None)) is not None:
            attrs[RESPONSE_FINISH_REASONS] = [stop_reason]
        if attrs:
            self._current_span.set_attributes(attrs)

        error = getattr(message, 'error', None)
        if error:  # pragma: no cover
            self._current_span.set_attribute(ERROR_TYPE, str(error))
            self._current_span.set_level('error')

    def close(self) -> None:
        """Close chat span and end any orphaned tool spans.

        Also finalises the root span with ``pydantic_ai.all_messages`` (the
        attribute that triggers logfire's "Agent Run" UI rendering on the
        root span) and ``claude.tools_used`` (per-tool invocation counts).
        """
        self.close_chat_span()
        # Snapshot under anyio: hooks can append to _history concurrently, so
        # iterate a copy rather than the live list to avoid a theoretical race
        # (list.append is safe; our walk is not atomic across the two reads).
        history_snapshot = list(self._history)
        if history_snapshot:
            self.root_span.set_attribute(PYDANTIC_AI_ALL_MESSAGES, history_snapshot)
            tool_counts = Counter(
                name
                for msg in history_snapshot
                for part in msg.get('parts', [])
                if part.get('type') == 'tool_call' and (name := part.get('name'))
            )
            if tool_counts:
                self.root_span.set_attribute(
                    CLAUDE_TOOLS_USED,
                    [{'tool': n, 'count': c} for n, c in tool_counts.items()],
                )
        # Hook-event counters: emit only the non-zero ones so happy-path
        # spans don't carry always-zero attributes.
        counter_map: tuple[tuple[int, str], ...] = (
            (self.user_prompt_count, CLAUDE_USER_PROMPT_COUNT),
            (self.compact_count, CLAUDE_COMPACT_COUNT),
            (self.notification_count, CLAUDE_NOTIFICATION_COUNT),
            (self.permission_request_count, CLAUDE_PERMISSION_REQUEST_COUNT),
        )
        for value, attr in counter_map:
            if value:
                self.root_span.set_attribute(attr, value)
        for span in self.active_tool_spans.values():
            span._end()  # pyright: ignore[reportPrivateUsage]
        self.active_tool_spans.clear()


def _record_rate_limit_event(
    logfire_instance: Logfire, root_span: LogfireSpan, msg: _RateLimitEvent
) -> None:
    """Record a RateLimitEvent as a level-appropriate log under the root span.

    The SDK emits these whenever the CLI reports a rate-limit state transition
    (``allowed`` → ``allowed_warning`` → ``rejected``), plus overage state. We
    surface them as logfire events so alerting can key off them.
    """
    info = getattr(msg, 'rate_limit_info', None)
    status = getattr(info, 'status', None)
    attrs: dict[str, Any] = {RATE_LIMIT_STATUS: status}
    # Map SDK field → semconv constant. Only emit set values.
    field_map: tuple[tuple[str, str], ...] = (
        ('resets_at', RATE_LIMIT_RESETS_AT),
        ('rate_limit_type', RATE_LIMIT_TYPE),
        ('utilization', RATE_LIMIT_UTILIZATION),
        ('overage_status', RATE_LIMIT_OVERAGE_STATUS),
        ('overage_resets_at', RATE_LIMIT_OVERAGE_RESETS_AT),
        ('overage_disabled_reason', RATE_LIMIT_OVERAGE_DISABLED_REASON),
        ('raw', RATE_LIMIT_RAW),
    )
    for sdk_field, attr_name in field_map:
        val = getattr(info, sdk_field, None)
        if val:
            attrs[attr_name] = val
    conv_id = getattr(msg, 'session_id', None)
    if conv_id is not None:
        attrs[CONVERSATION_ID] = conv_id

    if status == 'rejected':
        attrs[ERROR_TYPE] = 'RateLimitRejected'
        root_span.set_level('error')
        log = logfire_instance.error
    elif status == 'allowed_warning':
        log = logfire_instance.warn
    else:
        log = logfire_instance.info
    log('rate_limit {status}', status=status, **attrs)


def _record_mirror_error(logfire_instance: Logfire, msg: _MirrorErrorMessage) -> None:
    """Record a MirrorErrorMessage as an error-level log under the root span.

    Mirror errors are non-fatal (the local transcript is still durable) but
    indicate drift from an external SessionStore — worth surfacing. The raw
    SessionKey dict is unpacked into individual attributes (``session_id`` →
    ``gen_ai.conversation.id``, plus ``project_key`` / ``subpath``) rather
    than serialized whole, because logfire's default scrubber matches the
    substring ``session`` in attribute values and would redact the whole blob.
    """
    error = getattr(msg, 'error', '') or ''
    attrs: dict[str, Any] = {ERROR_TYPE: 'MirrorError'}
    key = getattr(msg, 'key', None)
    if isinstance(key, dict):
        if (sid := key.get('session_id')) is not None:
            attrs[CONVERSATION_ID] = sid
        if (pk := key.get('project_key')) is not None:
            attrs['mirror.project_key'] = pk
        if (sp := key.get('subpath')) is not None:
            attrs['mirror.subpath'] = sp
    logfire_instance.error('mirror store error: {error}', error=error, **attrs)


def _record_result(span: LogfireSpan, msg: ResultMessage) -> None:
    """Record ResultMessage data onto the root span."""
    attrs: dict[str, Any] = {}

    if hasattr(msg, 'usage') and msg.usage:  # pragma: no branch
        attrs.update(_extract_usage(msg.usage))

    if hasattr(msg, 'total_cost_usd') and msg.total_cost_usd is not None:  # pragma: no branch
        attrs['operation.cost'] = float(msg.total_cost_usd)

    session_id = getattr(msg, 'session_id', None)
    if session_id is not None:  # pragma: no branch
        attrs[CONVERSATION_ID] = session_id

    # Pass-through if not None (preserves empty string / empty dict for
    # ``claude.result.text`` and ``claude.result.structured_output``). These
    # claude.* attributes are added to ``scrubbing.SAFE_KEYS`` so model-
    # generated content and user-supplied schemas are not regex-scrubbed.
    for src, dst in (
        ('num_turns', 'num_turns'),
        ('duration_ms', 'duration_ms'),
        ('duration_api_ms', 'duration_api_ms'),
        ('subtype', CLAUDE_RESULT_SUBTYPE),
        ('result', CLAUDE_RESULT_TEXT),
        ('structured_output', CLAUDE_RESULT_STRUCTURED_OUTPUT),
    ):
        if (value := getattr(msg, src, None)) is not None:
            attrs[dst] = value

    # OTel semconv: finish_reasons is an array even with a single value.
    if (stop_reason := getattr(msg, 'stop_reason', None)) is not None:
        attrs[RESPONSE_FINISH_REASONS] = [stop_reason]

    # Per-model usage breakdown. Useful when fallback_model kicks in and
    # aggregate gen_ai.usage.* no longer attributes tokens to one model.
    if model_usage := getattr(msg, 'model_usage', None):
        attrs[CLAUDE_MODEL_USAGE] = model_usage

    # Truthy guards skip the empty-list happy path for these audit fields.
    # Note: claude.permission_denials is NOT scrub-exempt — denial entries
    # carry user-supplied ``tool_input`` where redaction is the safer default.
    if denials := getattr(msg, 'permission_denials', None):
        attrs[CLAUDE_PERMISSION_DENIALS] = denials

    if errors := getattr(msg, 'errors', None):
        attrs[CLAUDE_RESULT_ERRORS] = errors

    span.set_attributes(attrs)

    is_error = getattr(msg, 'is_error', None)
    if is_error:
        span.set_level('error')


# ---------------------------------------------------------------------------
# Hook-event emission helpers (issue #9). Each emits a level-appropriate
# logfire log under the active ``invoke_agent`` span — mirroring the
# precedent set by ``_record_rate_limit_event`` / ``_record_mirror_error``.
# Counters live on ``_ConversationState`` and are surfaced on the root span
# when the conversation closes.
# ---------------------------------------------------------------------------


def _hook_session_id(input_data: Any) -> dict[str, Any]:
    """Pull ``session_id`` from a hook input dict, mapped to the OTel semconv
    ``gen_ai.conversation.id``. Returns ``{}`` when absent so callers can
    splat it into a log call unconditionally.
    """
    sid = (input_data or {}).get('session_id') if isinstance(input_data, dict) else None
    return {CONVERSATION_ID: sid} if sid else {}


def _record_user_prompt_submit(state: _ConversationState, input_data: Any) -> None:
    """Record a UserPromptSubmit hook event as an info log under the root span.

    ``claude.user_prompt`` is intentionally NOT on the scrubber's SAFE_KEYS:
    user-typed prompts may contain credentials and default value-level
    redaction is the safer privacy posture. Operators who want raw prompts
    can use a ``scrubbing_callback``.
    """
    state.user_prompt_count += 1
    if not isinstance(input_data, dict):  # pragma: no cover
        return
    prompt = input_data.get('prompt')
    attrs: dict[str, Any] = {**_hook_session_id(input_data)}
    if prompt is not None:
        attrs[CLAUDE_USER_PROMPT] = prompt
    state.logfire.info('Hook: UserPromptSubmit', **attrs)


def _record_stop(state: _ConversationState, input_data: Any) -> None:
    """Record a Stop hook event as an info log under the root span.

    Captures ``last_assistant_message`` defensively via ``.get()`` —
    the field is on the CLI wire format but absent from the SDK's
    ``StopHookInput`` type, so older SDKs (or future renames) silently
    skip the attribute. Goes through the scrubber's SAFE_KEYS allowlist
    because it's model-generated text (mirrors ``claude.result.text``).
    """
    if not isinstance(input_data, dict):  # pragma: no cover
        return
    attrs: dict[str, Any] = {**_hook_session_id(input_data)}
    # Emit ``stop_hook_active`` only when True — almost always False.
    if input_data.get('stop_hook_active'):
        attrs[CLAUDE_STOP_HOOK_ACTIVE] = True
    last = input_data.get('last_assistant_message')
    if last is not None:
        attrs[CLAUDE_STOP_LAST_ASSISTANT_MESSAGE] = last
    state.logfire.info('Hook: Stop', **attrs)


def _record_pre_compact(state: _ConversationState, input_data: Any) -> None:
    """Record a PreCompact hook event as a warn log under the root span.

    Compaction is the strongest leading indicator that the session is
    approaching context limits; warn level keeps it filterable / alertable
    without being treated as a hard error.
    """
    state.compact_count += 1
    if not isinstance(input_data, dict):  # pragma: no cover
        return
    attrs: dict[str, Any] = {**_hook_session_id(input_data)}
    if (trigger := input_data.get('trigger')) is not None:
        attrs[CLAUDE_COMPACT_TRIGGER] = trigger
    if (instructions := input_data.get('custom_instructions')) is not None:
        attrs[CLAUDE_COMPACT_INSTRUCTIONS] = instructions
    state.logfire.warn('Hook: PreCompact ({trigger})', trigger=attrs.get(CLAUDE_COMPACT_TRIGGER, ''), **attrs)


def _record_notification(state: _ConversationState, input_data: Any) -> None:
    """Record a Notification hook event as an info log under the root span.

    ``claude.notification.message`` is NOT on SAFE_KEYS: CLI-emitted text
    can contain paths/words triggering false-positive scrubbing, and
    accepting that is preferable to allowlisting CLI-controlled content
    that may evolve.
    """
    state.notification_count += 1
    if not isinstance(input_data, dict):  # pragma: no cover
        return
    attrs: dict[str, Any] = {**_hook_session_id(input_data)}
    field_map: tuple[tuple[str, str], ...] = (
        ('message', CLAUDE_NOTIFICATION_MESSAGE),
        ('title', CLAUDE_NOTIFICATION_TITLE),
        ('notification_type', CLAUDE_NOTIFICATION_TYPE),
    )
    for sdk_field, attr_name in field_map:
        if (value := input_data.get(sdk_field)) is not None:
            attrs[attr_name] = value
    state.logfire.info(
        'Hook: Notification ({notification_type})',
        notification_type=attrs.get(CLAUDE_NOTIFICATION_TYPE, ''),
        **attrs,
    )


def _record_permission_request(state: _ConversationState, input_data: Any) -> None:
    """Record a PermissionRequest hook event as an info-level audit log.

    Captures ``agent_id`` / ``agent_type`` opportunistically when the
    request originates inside a subagent context (groundwork for #3).
    ``permission_suggestions`` shape is uncertain (``list[Any]``) — capture
    when truthy without imposing a sub-schema we'd have to break later.
    ``tool_input`` is caller-supplied arbitrary user data — deliberately
    NOT on SAFE_KEYS, matching the ``claude.permission_denials`` precedent.
    """
    state.permission_request_count += 1
    if not isinstance(input_data, dict):  # pragma: no cover
        return
    attrs: dict[str, Any] = {**_hook_session_id(input_data)}
    field_map: tuple[tuple[str, str], ...] = (
        ('tool_name', TOOL_NAME),
        ('tool_input', CLAUDE_PERMISSION_REQUEST_TOOL_INPUT),
        ('agent_id', CLAUDE_AGENT_ID),
        ('agent_type', CLAUDE_AGENT_TYPE),
    )
    for sdk_field, attr_name in field_map:
        if (value := input_data.get(sdk_field)) is not None:
            attrs[attr_name] = value
    # ``permission_suggestions`` shape is uncertain (``list[Any]``) — capture
    # only when truthy so we don't emit empty lists.
    if suggestions := input_data.get('permission_suggestions'):
        attrs[CLAUDE_PERMISSION_REQUEST_SUGGESTIONS] = suggestions
    state.logfire.info(
        'Hook: PermissionRequest ({tool_name})',
        tool_name=attrs.get(TOOL_NAME, ''),
        **attrs,
    )


def _record_permission_result(
    state: _ConversationState,
    tool_name: str,
    tool_use_id: str | None,
    result: Any,
) -> None:
    """Record a ``can_use_tool`` outcome (issue #10).

    Signature deliberately diverges from the issue-#9 ``_record_*`` helpers'
    ``(state, input_data)`` shape — this helper is invoked from
    ``_wrap_can_use_tool`` (a non-hook async callback wrapper, not a hook
    callback) and the source data is a typed ``PermissionResult`` dataclass
    rather than a hook JSON dict. Mapping the dataclass into ``input_data``
    just to fit the convention would obscure the typed contract.

    Pairs with the request-side ``Hook: PermissionRequest`` log from #9 —
    this captures what the user's ``can_use_tool`` callback decided, not
    just that the CLI asked. Shape:

    - **Allow** → info log; surfaces ``updated_input`` / ``updated_permissions``
      only when the callback mutated either, so happy-path Allow logs stay
      attribute-free beyond the behavior + tool_name.
    - **Deny** → warn log; surfaces ``message`` and ``interrupt`` (when True).
      Also escalates the still-open ``execute_tool`` span (the existing
      ``pre_tool_use_hook`` opens it before ``can_use_tool`` fires; deny
      aborts execution so ``post_tool_use_hook`` never runs and
      ``state.close()`` would otherwise end the span as a silent orphan).
      The level escalation + ``error.type='PermissionDeny'`` keeps the
      denied call visible mid-conversation, not just in the
      session-aggregate ``claude.permission_denials``.

    Per-attribute SAFE_KEYS rationale: see semconv.py — none of the new
    ``claude.permission_result.*`` attrs are allowlisted; default scrubbing
    is the safer posture for caller-supplied data.
    """
    behavior = getattr(result, 'behavior', None)
    attrs: dict[str, Any] = {
        TOOL_NAME: tool_name,
        CLAUDE_PERMISSION_RESULT_BEHAVIOR: behavior or 'unknown',
    }
    if tool_use_id is not None:
        attrs[TOOL_CALL_ID] = tool_use_id

    if isinstance(result, claude_agent_sdk.PermissionResultDeny):
        message = getattr(result, 'message', '')
        if message:
            attrs[CLAUDE_PERMISSION_RESULT_MESSAGE] = message
        if getattr(result, 'interrupt', False):
            attrs[CLAUDE_PERMISSION_RESULT_INTERRUPT] = True
        # Escalate AND immediately end the open ``execute_tool`` span. Deny
        # aborts execution before PostToolUse fires, so without ending the
        # span here it would stay open until ``state.close()`` runs at
        # end-of-conversation — producing a misleading ``end_time`` /
        # ``duration_ms`` that spans the entire remaining session for
        # every denied call.
        if tool_use_id is not None and (open_span := state.active_tool_spans.pop(tool_use_id, None)):
            open_span.set_attribute(ERROR_TYPE, 'PermissionDeny')
            if message:
                open_span.set_attribute(CLAUDE_PERMISSION_RESULT_MESSAGE, message)
            open_span.set_level('warn')
            open_span._end()  # pyright: ignore[reportPrivateUsage]
            # Also drop the diff snapshot — no PostToolUse will arrive to
            # consume it, and leaving it would leak across sessions.
            state.original_tool_inputs.pop(tool_use_id, None)
        state.logfire.warn('Hook: PermissionResult ({behavior})', behavior='deny', **attrs)
        return

    if isinstance(result, claude_agent_sdk.PermissionResultAllow):
        if (updated_input := getattr(result, 'updated_input', None)) is not None:
            attrs[CLAUDE_PERMISSION_RESULT_UPDATED_INPUT] = updated_input
        if (updated_permissions := getattr(result, 'updated_permissions', None)) is not None:
            # ``updated_permissions`` is a list of PermissionUpdate dataclasses;
            # serialise via ``.to_dict()`` (mirrors what the SDK does at the
            # wire boundary in query.py).
            try:
                attrs[CLAUDE_PERMISSION_RESULT_UPDATED_PERMISSIONS] = [
                    p.to_dict() if hasattr(p, 'to_dict') else p
                    for p in updated_permissions
                ]
            except Exception:  # pragma: no cover  # forward-compat: skip bad shape
                pass
        state.logfire.info('Hook: PermissionResult ({behavior})', behavior='allow', **attrs)
        return

    # Unknown PermissionResult subtype — log defensively under warn so a
    # future SDK change doesn't silently lose audit signal.
    state.logfire.warn('Hook: PermissionResult ({behavior})', behavior='unknown', **attrs)


def _wrap_can_use_tool(callback: Any) -> Any:
    """Wrap a user-supplied ``can_use_tool`` callback so each invocation is
    surfaced as a logfire log under the active ``invoke_agent`` span.

    Idempotent: calling on an already-wrapped callable is a no-op (returns
    the same wrapper). Used by ``_inject_tracing_hooks`` to avoid double-
    wrap when the same options object is reused across multiple
    ``ClaudeSDKClient`` instances.
    """
    if getattr(callback, '_logfire_wrapped', False):
        return callback

    @functools.wraps(callback)
    async def wrapped(tool_name: str, tool_input: Any, context: Any) -> Any:
        result = await callback(tool_name, tool_input, context)
        with handle_internal_errors:
            state = _get_state()
            if state is None:  # pragma: no cover
                return result
            token = _attach_root_context(state)
            if token is None:  # pragma: no cover
                return result
            try:
                _record_permission_result(
                    state,
                    tool_name=tool_name,
                    tool_use_id=getattr(context, 'tool_use_id', None),
                    result=result,
                )
            finally:
                context_api.detach(token)
        return result

    wrapped._logfire_wrapped = True  # type: ignore[attr-defined]
    return wrapped
