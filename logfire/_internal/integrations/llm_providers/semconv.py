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
