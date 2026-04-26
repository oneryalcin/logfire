"""Gen AI Semantic Convention attribute names and type definitions.

These constants and types follow the OpenTelemetry Gen AI Semantic Conventions.
See: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-events/
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Union

from typing_extensions import NotRequired, TypeAlias, TypedDict

# Version type for controlling span attribute format
SemconvVersion = Literal[1, 'latest']


ALLOWED_VERSIONS: frozenset[SemconvVersion] = frozenset((1, 'latest'))


def normalize_versions(version: SemconvVersion | Sequence[SemconvVersion]) -> frozenset[SemconvVersion]:
    """Normalize a version parameter to a validated frozenset of version values."""
    if isinstance(version, (int, str)):
        versions: frozenset[Any] = frozenset({version})
    else:
        versions = frozenset(version)

    invalid = versions - ALLOWED_VERSIONS
    if invalid:
        raise ValueError(
            f"Invalid semconv version(s): {sorted(invalid, key=repr)!r}. Supported versions are: 1, 'latest'."
        )

    if not versions:
        raise ValueError("At least one semconv version must be specified. Supported versions are: 1, 'latest'.")

    return versions


# Provider, system, and operation
PROVIDER_NAME = 'gen_ai.provider.name'
SYSTEM = 'gen_ai.system'
OPERATION_NAME = 'gen_ai.operation.name'


def provider_attrs(name: str) -> dict[str, str]:
    """Return the common {SYSTEM: name, PROVIDER_NAME: name} dict."""
    return {SYSTEM: name, PROVIDER_NAME: name}


# Model information
REQUEST_MODEL = 'gen_ai.request.model'
RESPONSE_MODEL = 'gen_ai.response.model'

# Request parameters
REQUEST_MAX_TOKENS = 'gen_ai.request.max_tokens'
REQUEST_TEMPERATURE = 'gen_ai.request.temperature'
REQUEST_TOP_P = 'gen_ai.request.top_p'
REQUEST_TOP_K = 'gen_ai.request.top_k'
REQUEST_STOP_SEQUENCES = 'gen_ai.request.stop_sequences'
REQUEST_SEED = 'gen_ai.request.seed'
REQUEST_FREQUENCY_PENALTY = 'gen_ai.request.frequency_penalty'
REQUEST_PRESENCE_PENALTY = 'gen_ai.request.presence_penalty'

# Response metadata
RESPONSE_ID = 'gen_ai.response.id'
RESPONSE_FINISH_REASONS = 'gen_ai.response.finish_reasons'

# Token usage
INPUT_TOKENS = 'gen_ai.usage.input_tokens'
OUTPUT_TOKENS = 'gen_ai.usage.output_tokens'
CACHE_READ_INPUT_TOKENS = 'gen_ai.usage.cache_read.input_tokens'
CACHE_CREATION_INPUT_TOKENS = 'gen_ai.usage.cache_creation.input_tokens'
USAGE_RAW = 'gen_ai.usage.raw'

# Message content
INPUT_MESSAGES = 'gen_ai.input.messages'
OUTPUT_MESSAGES = 'gen_ai.output.messages'
SYSTEM_INSTRUCTIONS = 'gen_ai.system_instructions'
# Full conversation history on the agent root span. Logfire's Model Run UI
# renders this specific attribute as the root-span chat view.
PYDANTIC_AI_ALL_MESSAGES = 'pydantic_ai.all_messages'

# Tool execution
TOOL_DEFINITIONS = 'gen_ai.tool.definitions'
TOOL_NAME = 'gen_ai.tool.name'
TOOL_CALL_ID = 'gen_ai.tool.call.id'
TOOL_CALL_ARGUMENTS = 'gen_ai.tool.call.arguments'
TOOL_CALL_RESULT = 'gen_ai.tool.call.result'

# Conversation tracking
CONVERSATION_ID = 'gen_ai.conversation.id'

# OTel GenAI semconv attribute identifying the agent framework.
AGENT_NAME = 'gen_ai.agent.name'

# Working directory the Claude Agent SDK is running in. Intentionally under
# the vendor-prefixed ``claude.*`` namespace (rather than the ``session.*``
# convention used by some Claude-Code-family plugins) so value-level scrubbing
# still runs against the path — cwd strings legitimately contain substrings
# like ``auth``, ``secret``, ``private_keys``, etc., and regex scrubbing there
# is the safer default.
CLAUDE_CWD = 'claude.cwd'

# Error
ERROR_TYPE = 'error.type'

# Rate-limit (custom extension; not yet in upstream OTel gen-ai semconv).
# Surfaced by the Claude Agent SDK as RateLimitEvent messages.
RATE_LIMIT_STATUS = 'gen_ai.rate_limit.status'
RATE_LIMIT_TYPE = 'gen_ai.rate_limit.type'
RATE_LIMIT_UTILIZATION = 'gen_ai.rate_limit.utilization'
RATE_LIMIT_RESETS_AT = 'gen_ai.rate_limit.resets_at'
RATE_LIMIT_OVERAGE_STATUS = 'gen_ai.rate_limit.overage.status'
RATE_LIMIT_OVERAGE_RESETS_AT = 'gen_ai.rate_limit.overage.resets_at'
RATE_LIMIT_OVERAGE_DISABLED_REASON = 'gen_ai.rate_limit.overage.disabled_reason'
RATE_LIMIT_RAW = 'gen_ai.rate_limit.raw'

# Claude Agent SDK result-message fields (vendor-prefixed extensions; not in
# upstream OTel gen-ai semconv). Surfaced from ResultMessage which summarises
# the end-of-conversation state. The `text`, `errors`, `structured_output`,
# `subtype`, and `model_usage` keys are added to the scrubber's SAFE_KEYS
# allowlist — they carry model-generated content or deterministic enums/
# numbers, where default regex-based scrubbing does more harm than good.
# `permission_denials` is deliberately NOT on the allowlist: denial entries
# include the caller-supplied ``tool_input`` which is genuinely arbitrary
# user data where scrubbing is the safer default.
CLAUDE_RESULT_SUBTYPE = 'claude.result.subtype'
CLAUDE_RESULT_TEXT = 'claude.result.text'
CLAUDE_RESULT_ERRORS = 'claude.result.errors'
CLAUDE_RESULT_STRUCTURED_OUTPUT = 'claude.result.structured_output'
CLAUDE_MODEL_USAGE = 'claude.model_usage'
CLAUDE_PERMISSION_DENIALS = 'claude.permission_denials'
# Aggregated tool usage across the conversation, emitted on the
# ``invoke_agent`` root span at close for dashboards that group by tool.
CLAUDE_TOOLS_USED = 'claude.tools_used'

# SDK-internal per-AssistantMessage ids, distinct from the Anthropic API
# message_id captured under ``gen_ai.response.id``. ``parent_tool_use_id``
# stitches subagent responses back to the ToolUse that spawned them.
CLAUDE_MESSAGE_UUID = 'claude.message.uuid'
CLAUDE_PARENT_TOOL_USE_ID = 'claude.parent_tool_use_id'

# ClaudeAgentOptions-derived attributes surfaced on the ``invoke_agent``
# root span when the caller configured them. Names avoid the ``session``
# substring (which would trip the default scrubber): ``resume`` →
# ``resume_from``, ``fork_session`` → ``fork_on_resume``.
# ``model`` and ``fallback_model`` carry the deeper ``.options.`` prefix to
# disambiguate from per-turn ``gen_ai.request.model`` on chat spans. Other
# options-derived attrs below are bare ``claude.<field>`` because they have
# no collision in the existing namespace.
CLAUDE_OPTIONS_MODEL = 'claude.options.model'
CLAUDE_OPTIONS_FALLBACK_MODEL = 'claude.options.fallback_model'
CLAUDE_PERMISSION_MODE = 'claude.permission_mode'
CLAUDE_MAX_TURNS = 'claude.max_turns'
CLAUDE_MAX_BUDGET_USD = 'claude.max_budget_usd'
CLAUDE_ALLOWED_TOOLS = 'claude.allowed_tools'
CLAUDE_DISALLOWED_TOOLS = 'claude.disallowed_tools'
CLAUDE_EFFORT = 'claude.effort'
CLAUDE_AGENTS = 'claude.agents'
# ``skills`` on ClaudeAgentOptions is ``list[str] | Literal["all"] | None``.
# We split the mixed shape into a discriminator + list so downstream
# typed stores see one schema: ``claude.skills_mode`` is "all" or
# "allowlist"; ``claude.skills`` is the list (omitted in "all" mode).
CLAUDE_SKILLS_MODE = 'claude.skills_mode'
CLAUDE_SKILLS = 'claude.skills'
CLAUDE_SETTING_SOURCES = 'claude.setting_sources'
CLAUDE_CONTINUE_CONVERSATION = 'claude.continue_conversation'
CLAUDE_INCLUDE_PARTIAL_MESSAGES = 'claude.include_partial_messages'
CLAUDE_ENABLE_FILE_CHECKPOINTING = 'claude.enable_file_checkpointing'
CLAUDE_RESUME_FROM = 'claude.resume_from'
CLAUDE_FORK_ON_RESUME = 'claude.fork_on_resume'

# Hook-event attributes (issue #9). Each non-tool-lifecycle hook that the
# Claude Agent SDK fires (UserPromptSubmit / Stop / PreCompact / Notification
# / PermissionRequest) emits a level-appropriate logfire log under the
# active ``invoke_agent`` span. Cumulative counts are aggregated on the
# root span at conversation close (mirroring ``claude.tools_used``).
#
# Naming: counter attrs use a bare ``_count`` suffix (``claude.compact_count``)
# rather than nesting under the per-event sub-namespace (``claude.compact.count``)
# — they are root-span session aggregates, analogous to ``claude.tools_used``,
# not per-event detail attributes. Don't "fix" this asymmetry without
# updating the dashboards that key on these names.
#
# SAFE_KEYS additions: ``claude.compact.custom_instructions`` and
# ``claude.stop.last_assistant_message`` carry model-generated /
# operator-set text that default regex scrubbing would over-redact;
# ``claude.user_prompt`` deliberately stays subject to scrubbing because
# user-typed prompts may contain credentials. See scrubbing.py for the
# exact allowlist.
CLAUDE_USER_PROMPT = 'claude.user_prompt'
CLAUDE_USER_PROMPT_COUNT = 'claude.user_prompt_count'
CLAUDE_STOP_HOOK_ACTIVE = 'claude.stop.hook_active'
# ``last_assistant_message`` is on the wire but not in the SDK's StopHookInput
# type; capture defensively via .get().
CLAUDE_STOP_LAST_ASSISTANT_MESSAGE = 'claude.stop.last_assistant_message'
CLAUDE_COMPACT_TRIGGER = 'claude.compact.trigger'
CLAUDE_COMPACT_INSTRUCTIONS = 'claude.compact.custom_instructions'
CLAUDE_COMPACT_COUNT = 'claude.compact_count'
CLAUDE_NOTIFICATION_MESSAGE = 'claude.notification.message'
CLAUDE_NOTIFICATION_TITLE = 'claude.notification.title'
CLAUDE_NOTIFICATION_TYPE = 'claude.notification.type'
CLAUDE_NOTIFICATION_COUNT = 'claude.notification_count'
CLAUDE_PERMISSION_REQUEST_TOOL_INPUT = 'claude.permission_request.tool_input'
CLAUDE_PERMISSION_REQUEST_SUGGESTIONS = 'claude.permission_request.suggestions'
CLAUDE_PERMISSION_REQUEST_COUNT = 'claude.permission_request_count'
# Subagent attribution (also surfaced from PermissionRequest hooks fired
# inside a subagent context — groundwork for issue #3).
CLAUDE_AGENT_ID = 'claude.agent_id'
CLAUDE_AGENT_TYPE = 'claude.agent_type'

# Permission-flow attributes (issue #10). Pair with the request-side
# ``claude.permission_request.*`` from #9: the *outcome* lives here.
#
# - ``behavior`` is ``"allow"`` / ``"deny"`` (the SDK's PermissionResult enum).
# - ``message`` and ``interrupt`` are populated only on ``PermissionResultDeny``.
# - ``updated_input`` / ``updated_permissions`` are populated only on
#   ``PermissionResultAllow`` when the user callback mutated either.
#
# Deliberately NOT on SAFE_KEYS: ``updated_input`` is caller-supplied
# arbitrary data, ``message`` is operator-set (could echo any string).
# Default scrubbing is the safer privacy posture, mirroring
# ``claude.permission_denials`` and ``claude.permission_request.*``.
CLAUDE_PERMISSION_RESULT_BEHAVIOR = 'claude.permission_result.behavior'
CLAUDE_PERMISSION_RESULT_MESSAGE = 'claude.permission_result.message'
CLAUDE_PERMISSION_RESULT_INTERRUPT = 'claude.permission_result.interrupt'
CLAUDE_PERMISSION_RESULT_UPDATED_INPUT = 'claude.permission_result.updated_input'
CLAUDE_PERMISSION_RESULT_UPDATED_PERMISSIONS = 'claude.permission_result.updated_permissions'

# When a hook (PreToolUse output ``updatedInput`` or ``can_use_tool``'s
# ``PermissionResultAllow.updated_input``) mutates a tool's input between
# pre- and post-hook fire, the OTel-canonical ``gen_ai.tool.call.arguments``
# is overwritten with the executed (post) value and the original (pre)
# value lands here. Absent on the happy path where pre == executed.
CLAUDE_TOOL_CALL_ARGUMENTS_ORIGINAL = 'claude.tool_call.arguments.original'

# Lifecycle / control-method attributes (issue #11). Captured on the spans
# wrapping ``ClaudeSDKClient.connect`` / ``disconnect`` / ``set_model`` /
# ``set_permission_mode`` / ``rewind_files`` / ``reconnect_mcp_server`` /
# ``toggle_mcp_server`` / ``stop_task`` / ``interrupt``.
#
# None on SAFE_KEYS:
#   * ``claude.process.stderr`` and ``claude.cli_path`` can carry filesystem
#     paths matching default scrub patterns (``auth``, ``secret``,
#     ``private_key``, etc.) — value-level redaction is the safer default.
#   * The remaining attrs (``exit_code`` int, ``mcp.enabled`` bool,
#     ``mcp.server_name`` / ``task_id`` / ``rewind.user_message_id`` —
#     short operator-set identifiers) are not type-sensitive but are also
#     omitted from SAFE_KEYS for symmetry; default scrubbing is a no-op
#     on integers / booleans, and the identifier strings are unlikely
#     to legitimately contain credentials.
CLAUDE_CLI_PATH = 'claude.cli_path'
CLAUDE_PROCESS_EXIT_CODE = 'claude.process.exit_code'
CLAUDE_PROCESS_STDERR = 'claude.process.stderr'
CLAUDE_REWIND_USER_MESSAGE_ID = 'claude.rewind.user_message_id'
CLAUDE_MCP_SERVER_NAME = 'claude.mcp.server_name'
CLAUDE_MCP_ENABLED = 'claude.mcp.enabled'
CLAUDE_TASK_ID = 'claude.task_id'

# Subagent / Task* attributes (issue #3). Surfaced on the
# ``subagent <agent_type>`` span opened on ``SubagentStart`` and on logs
# dispatched from ``TaskStartedMessage`` / ``TaskProgressMessage`` /
# ``TaskNotificationMessage`` system-message events.
#
# SAFE_KEYS treatment (in scrubbing.py):
#   * ``claude.subagent.last_assistant_message`` — model-generated final
#     subagent text. Allowlisted: same shape as
#     ``claude.stop.last_assistant_message`` (the parent-thread analogue),
#     which is also allowlisted. Operators expect the verbatim text to
#     survive default scrubbing, since pattern-based redaction would
#     destroy model output content.
#   * ``claude.task.summary`` — model-generated end-of-task summary.
#     Allowlisted: same shape as ``claude.result.text`` (also allowlisted).
#
# NOT on SAFE_KEYS:
#   * ``claude.task.description`` — caller-supplied prompt text from the
#     parent's Task tool call (``ToolUseBlock.input.description``); default
#     scrubbing is the safer privacy posture, mirroring ``claude.user_prompt``.
#   * ``claude.agent.transcript_path`` — filesystem path that may match
#     ``auth``/``secret``/``private_key`` patterns; default scrub correct.
#   * Other fields are short identifiers / enums / int dicts where scrubbing
#     is a no-op or near-no-op.
CLAUDE_TASK_DESCRIPTION = 'claude.task.description'
CLAUDE_TASK_TYPE = 'claude.task.type'
CLAUDE_TASK_STATUS = 'claude.task.status'
CLAUDE_TASK_SUMMARY = 'claude.task.summary'
CLAUDE_TASK_OUTPUT_FILE = 'claude.task.output_file'
CLAUDE_TASK_USAGE = 'claude.task.usage'
CLAUDE_TASK_LAST_TOOL_NAME = 'claude.task.last_tool_name'
CLAUDE_AGENT_TRANSCRIPT_PATH = 'claude.agent.transcript_path'
# ``last_assistant_message`` is on the wire on ``SubagentStop`` events
# but NOT in the SDK's ``SubagentStopHookInput`` type — captured
# defensively via ``.get()`` (mirrors ``claude.stop.last_assistant_message``
# from #9).
CLAUDE_SUBAGENT_LAST_ASSISTANT_MESSAGE = 'claude.subagent.last_assistant_message'
# Cumulative subagent count emitted on the root invoke_agent at close.
CLAUDE_SUBAGENT_COUNT = 'claude.subagent_count'

# ``AgentDefinition`` metadata captured on the subagent span at
# ``SubagentStart`` time (issue #3, Option B). Mirrors the
# ``_options_attrs`` pattern from #8 — looked up via
# ``options.agents[agent_type]``. Without this, the subagent span shows
# only ``agent_id`` / ``agent_type`` / transcript path; with it, dashboards
# can group / filter by which subagent model / tool allowlist is in use,
# and the system prompt is queryable for audit.
CLAUDE_AGENT_MODEL = 'claude.agent.model'
CLAUDE_AGENT_DESCRIPTION = 'claude.agent.description'
CLAUDE_AGENT_TOOLS = 'claude.agent.tools'
CLAUDE_AGENT_DISALLOWED_TOOLS = 'claude.agent.disallowed_tools'
CLAUDE_AGENT_SKILLS = 'claude.agent.skills'
CLAUDE_AGENT_MEMORY = 'claude.agent.memory'
CLAUDE_AGENT_BACKGROUND = 'claude.agent.background'
# ``claude.agent.system_prompt`` and ``claude.agent.initial_prompt`` carry
# operator-set guidance text. Both added to SAFE_KEYS — same rationale as
# ``claude.compact.custom_instructions`` from #9: operator-controlled, not
# user-runtime input.
CLAUDE_AGENT_SYSTEM_PROMPT = 'claude.agent.system_prompt'
CLAUDE_AGENT_INITIAL_PROMPT = 'claude.agent.initial_prompt'
# Note: ``AgentDefinition.permissionMode`` / ``maxTurns`` / ``effort`` are
# captured under the existing ``CLAUDE_PERMISSION_MODE`` / ``CLAUDE_MAX_TURNS``
# / ``CLAUDE_EFFORT`` constants from #8 — same configuration concepts;
# span name (``invoke_agent`` vs ``subagent <agent_type>``) disambiguates
# whether they're parent-options or subagent-definition values in dashboards.
# ``mcpServers`` is deliberately not captured here: potentially large,
# contains nested config dicts; deferred to a separate extractor.

# Type definitions for message parts and messages


class TextPart(TypedDict):
    """Text content part."""

    type: Literal['text']
    content: str


class ToolCallPart(TypedDict):
    """Tool call part."""

    type: Literal['tool_call']
    id: str
    name: str
    arguments: NotRequired[dict[str, Any] | str | None]


class ToolCallResponsePart(TypedDict):
    """Tool call response part."""

    type: Literal['tool_call_response']
    id: str
    name: NotRequired[str]
    response: NotRequired[str | dict[str, Any] | None]


class UriPart(TypedDict):
    """URI-based media part (image, audio, video, document)."""

    type: Literal['uri']
    uri: str
    modality: NotRequired[Literal['image', 'audio', 'video', 'document']]


class BlobPart(TypedDict):
    """Binary data part."""

    type: Literal['blob']
    content: str
    media_type: NotRequired[str]
    modality: NotRequired[Literal['image', 'audio', 'video', 'document']]


class ReasoningPart(TypedDict):
    """Reasoning/thinking content part."""

    type: Literal['reasoning']
    content: str


MessagePart: TypeAlias = Union[
    TextPart, ToolCallPart, ToolCallResponsePart, UriPart, BlobPart, ReasoningPart, dict[str, Any]
]
"""A message part.

Can be any of the defined part types or a generic dict for extensibility.
"""


Role = Literal['system', 'user', 'assistant', 'tool']
"""Valid message roles."""


class ChatMessage(TypedDict):
    """A chat message following OTel Gen AI Semantic Conventions."""

    role: Role
    parts: list[MessagePart]
    name: NotRequired[str]


InputMessages: TypeAlias = list[ChatMessage]
"""List of input messages."""


SystemInstructions: TypeAlias = list[MessagePart]
"""System instructions as a list of message parts."""


class OutputMessage(ChatMessage):
    """An output message with optional finish reason."""

    finish_reason: NotRequired[str]


OutputMessages: TypeAlias = list[OutputMessage]
"""List of output messages."""
