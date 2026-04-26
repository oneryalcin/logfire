# pyright: reportPrivateUsage=false
"""Tests for Claude Agent SDK instrumentation.

Unit tests for helper functions (content block conversion, usage extraction,
hook functions, hook injection) plus cassette-based integration tests that
replay recorded sessions through the real SubprocessCLITransport.

Recording cassettes (requires a real `claude` CLI with valid credentials):
    uv run pytest tests/otel_integrations/test_claude_agent_sdk.py --record-claude-cassettes

Replaying (default, no real CLI needed):
    uv run pytest tests/otel_integrations/test_claude_agent_sdk.py
"""

from __future__ import annotations

import os
import shutil
import stat
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

pytest.importorskip('claude_agent_sdk', reason='claude_agent_sdk requires Python 3.10+')

from claude_agent_sdk import (
    AgentDefinition,
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    PermissionResultAllow,
    PermissionResultDeny,
    ServerToolResultBlock,
    ServerToolUseBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from claude_agent_sdk._errors import CLINotFoundError, ProcessError
from claude_agent_sdk.types import HookContext, HookEvent
from dirty_equals import IsStr
from inline_snapshot import snapshot

import logfire
from logfire._internal.integrations.claude_agent_sdk import (
    _annotate_error_span,
    _clear_state,
    _content_blocks_to_output_messages,
    _ConversationState,
    _extract_usage,
    _inject_tracing_hooks,
    _options_attrs,
    _record_mirror_error,
    _record_notification,
    _record_permission_request,
    _record_permission_result,
    _record_pre_compact,
    _record_rate_limit_event,
    _record_result,
    _record_stop,
    _record_subagent_start,
    _record_subagent_stop,
    _record_task_notification,
    _record_task_progress,
    _record_task_started,
    _record_user_prompt_submit,
    _set_state,
    _subagent_definition_attrs,
    _tool_parent_otel_span,
    _wrap_can_use_tool,
    notification_hook,
    permission_request_hook,
    post_tool_use_failure_hook,
    post_tool_use_hook,
    pre_compact_hook,
    pre_tool_use_hook,
    stop_hook,
    user_prompt_submit_hook,
)
from logfire.testing import TestExporter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FAKE_CLAUDE = Path(__file__).parent / 'fake_claude.py'
CASSETTES_DIR = Path(__file__).parent / 'cassettes' / 'test_claude_agent_sdk'

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_instrumentation():  # pyright: ignore[reportUnusedFunction]
    """Instrument and reset SDK class patching between tests."""
    with logfire.instrument_claude_agent_sdk():
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _close_sdk_streams(client: ClaudeSDKClient) -> None:
    """Close streams the SDK neglects to close, preventing ResourceWarning on GC.

    The SDK's Query.close() doesn't close its internal MemoryObject streams,
    and SubprocessCLITransport.close() doesn't close stdout. Close them
    explicitly before disconnect() sets _query/_transport to None.
    """
    query = client._query
    if query is not None:
        # Closing a query may use the streams, so that has to be done first.
        with suppress(Exception):
            await query.close()
        with suppress(Exception):
            await query._message_send.aclose()
        with suppress(Exception):
            await query._message_receive.aclose()
        # Disconnecting the client tries to close the query again if it's not None, and it can't.
        client._query = None
    transport = client._transport
    if transport is not None:
        stdout = getattr(transport, '_stdout_stream', None)
        if stdout is not None:
            with suppress(Exception):
                await stdout.aclose()


def _make_client(
    cassette_name: str,
    *,
    monkeypatch: pytest.MonkeyPatch,
    system_prompt: str = 'Be helpful',
    record: bool = False,
) -> ClaudeSDKClient:
    """Create a ClaudeSDKClient backed by a cassette file.

    In replay mode (default), uses fake_claude.py to replay a recorded session.
    In record mode, uses fake_claude.py as a proxy to the real claude CLI,
    recording the session to the cassette file.
    """
    cassette_path = CASSETTES_DIR / cassette_name

    if not record and not cassette_path.exists():
        raise FileNotFoundError(
            f'Cassette not found: {cassette_path}\n'
            f'Record it with: uv run pytest {__file__} --record-claude-cassettes -k <test_name>'
        )

    # Ensure fake_claude.py is executable
    fake_claude_path = str(FAKE_CLAUDE)
    st = os.stat(fake_claude_path)
    if not (st.st_mode & stat.S_IEXEC):
        os.chmod(fake_claude_path, st.st_mode | stat.S_IEXEC)

    monkeypatch.setenv('CASSETTE_PATH', str(cassette_path))

    if record:
        real_claude = shutil.which('claude')
        if not real_claude:
            pytest.skip('Real claude CLI not found on PATH; cannot record cassette')
        monkeypatch.setenv('CASSETTE_MODE', 'record')
        monkeypatch.setenv('REAL_CLAUDE_PATH', real_claude)
    else:
        monkeypatch.setenv('CASSETTE_MODE', 'replay')
        monkeypatch.delenv('REAL_CLAUDE_PATH', raising=False)

    monkeypatch.setenv('CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK', '1')

    return ClaudeSDKClient(
        options=ClaudeAgentOptions(
            system_prompt=system_prompt,
            cli_path=fake_claude_path,
        ),
    )


# ---------------------------------------------------------------------------
# Utility function tests (pure unit tests, no SDK dependency).
# ---------------------------------------------------------------------------


def test_content_blocks_to_output_messages() -> None:
    # Non-list returns empty
    assert _content_blocks_to_output_messages('just a string') == []

    # TextBlock
    assert _content_blocks_to_output_messages([TextBlock(text='hello world')]) == [
        {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'hello world'}]}
    ]

    # ThinkingBlock → ReasoningPart
    assert _content_blocks_to_output_messages([ThinkingBlock(thinking='let me think...', signature='sig123')]) == [
        {'role': 'assistant', 'parts': [{'type': 'reasoning', 'content': 'let me think...'}]}
    ]

    # ToolUseBlock
    assert _content_blocks_to_output_messages([ToolUseBlock(id='tool_1', name='Bash', input={'command': 'ls'})]) == [
        {
            'role': 'assistant',
            'parts': [{'type': 'tool_call', 'id': 'tool_1', 'name': 'Bash', 'arguments': {'command': 'ls'}}],
        }
    ]

    # ToolResultBlock — content passed through directly
    assert _content_blocks_to_output_messages([ToolResultBlock(tool_use_id='tool_1', content='output text')]) == [
        {'role': 'assistant', 'parts': [{'type': 'tool_call_response', 'id': 'tool_1', 'response': 'output text'}]}
    ]

    # ServerToolUseBlock — same shape as ToolUseBlock. `name` is the
    # server-tool discriminator (web_search, advisor, code_execution, ...).
    assert _content_blocks_to_output_messages(
        [ServerToolUseBlock(id='srv_1', name='web_search', input={'query': 'OTel gen_ai'})]
    ) == [
        {
            'role': 'assistant',
            'parts': [
                {'type': 'tool_call', 'id': 'srv_1', 'name': 'web_search', 'arguments': {'query': 'OTel gen_ai'}}
            ],
        }
    ]

    # ServerToolResultBlock — content is a dict; passed through unchanged.
    assert _content_blocks_to_output_messages(
        [ServerToolResultBlock(tool_use_id='srv_1', content={'type': 'advisor_tool_result', 'summary': 'ok'})]
    ) == [
        {
            'role': 'assistant',
            'parts': [
                {
                    'type': 'tool_call_response',
                    'id': 'srv_1',
                    'response': {'type': 'advisor_tool_result', 'summary': 'ok'},
                }
            ],
        }
    ]

    # Unknown block type passes through
    block = Mock()
    result = _content_blocks_to_output_messages([block])
    assert len(result) == 1
    assert result[0]['parts'][0] is block


@pytest.mark.anyio
async def test_record_result_skips_empty_and_none_fields(exporter: TestExporter) -> None:
    """``_record_result`` should only set attributes for fields with meaningful
    values: empty lists (``errors=[]``, ``permission_denials=[]``) and
    ``None`` values are skipped so happy-path spans don't carry always-empty
    attributes.

    Locked semantics:
      * ``errors=[]``        → no ``claude.result.errors`` attribute
      * ``permission_denials=[]`` → no ``claude.permission_denials``
      * ``stop_reason=None`` → no ``gen_ai.response.finish_reasons``
      * ``result=None``      → no ``claude.result.text``
      * ``structured_output=None`` → no ``claude.result.structured_output``
      * ``model_usage=None`` → no ``claude.model_usage``
    """
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent') as root:
        msg = SimpleNamespace(
            usage={'input_tokens': 100, 'output_tokens': 50},
            total_cost_usd=0.01,
            session_id='sess-abc',
            num_turns=1,
            duration_ms=1000,
            duration_api_ms=900,
            subtype='success',
            stop_reason=None,
            model_usage=None,
            permission_denials=[],
            errors=[],
            result=None,
            structured_output=None,
            is_error=False,
        )
        _record_result(root, msg)  # type: ignore[arg-type]

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    root_attrs = next(s for s in spans if s['name'] == 'invoke_agent')['attributes']

    # Populated fields are present.
    assert root_attrs['claude.result.subtype'] == 'success'
    assert root_attrs['duration_api_ms'] == 900
    assert root_attrs['num_turns'] == 1

    # Empty / None fields are skipped.
    for absent in (
        'gen_ai.response.finish_reasons',
        'claude.result.text',
        'claude.model_usage',
        'claude.permission_denials',
        'claude.result.errors',
        'claude.result.structured_output',
    ):
        assert absent not in root_attrs, f'{absent!r} should be skipped when empty/None'


@pytest.mark.anyio
async def test_record_result_is_error_escalates_span_level(exporter: TestExporter) -> None:
    """A ResultMessage with ``is_error=True`` escalates the root span to error."""
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent') as root:
        msg = SimpleNamespace(
            usage={'input_tokens': 10, 'output_tokens': 0},
            total_cost_usd=0.0,
            session_id='sess-xyz',
            num_turns=3,
            duration_ms=500,
            duration_api_ms=400,
            subtype='error_max_turns',
            stop_reason=None,
            model_usage=None,
            permission_denials=[],
            errors=['hit max_turns cap'],
            result=None,
            structured_output=None,
            is_error=True,
        )
        _record_result(root, msg)  # type: ignore[arg-type]

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    root_span = next(s for s in spans if s['name'] == 'invoke_agent')
    assert root_span['attributes']['logfire.level_num'] == 17  # error
    assert root_span['attributes']['claude.result.subtype'] == 'error_max_turns'
    assert root_span['attributes']['claude.result.errors'] == ['hit max_turns cap']


def _user_msg(text: str) -> Any:
    """Test helper — build a minimal ``gen_ai.input.messages``-shaped user message."""
    return {'role': 'user', 'parts': [{'type': 'text', 'content': text}]}


def _drive_state_through_turns(state: _ConversationState, turns: list[AssistantMessage]) -> None:
    """Drive ``_ConversationState`` through assistant turns via the public
    ``handle_assistant_message`` / ``handle_user_message`` / ``close``
    API — no reaching into ``_history`` / ``_current_output_parts``.

    The first turn uses the already-open chat span from ``open_chat_span()``;
    subsequent turns call ``handle_user_message()`` to close the prior span
    and open a new one (the same state-machine trigger the real receive loop
    uses when a UserMessage arrives in the stream).
    """
    state.open_chat_span()
    for i, turn in enumerate(turns):
        if i > 0:
            state.handle_user_message()
        state.handle_assistant_message(turn)


@pytest.mark.anyio
async def test_close_sets_pydantic_ai_all_messages(exporter: TestExporter) -> None:
    """``_ConversationState.close()`` copies ``_history`` onto the root span
    as ``pydantic_ai.all_messages``.

    Locked as a test because this attribute is what flips logfire's UI
    from generic-span rendering to the full "Agent Run" view (full
    conversation on the root span). A regression that empties ``_history``
    or renames the attribute silently removes the root-span UI treatment
    without breaking any chat-span-level snapshot.
    """
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[_user_msg('hi')])
        _drive_state_through_turns(
            state,
            [AssistantMessage(content=[TextBlock(text='hello')], model='claude-sonnet-4-6')],
        )
        state.close()

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    root_attrs = next(s for s in spans if s['name'] == 'invoke_agent')['attributes']
    all_messages = root_attrs['pydantic_ai.all_messages']
    assert [m['role'] for m in all_messages] == ['user', 'assistant']
    assert all_messages[0]['parts'][0]['content'] == 'hi'
    assert all_messages[1]['parts'][0]['content'] == 'hello'


@pytest.mark.anyio
async def test_close_aggregates_tools_used_from_history(exporter: TestExporter) -> None:
    """``_ConversationState.close()`` walks ``_history`` and emits a
    ``claude.tools_used`` aggregate count per tool on the root span.

    Covers client tools, MCP tools, and server tools uniformly — they all
    appear as ``type='tool_call'`` parts in history, so one pass counts all.
    Mixed-role: also adds a ``role='tool'`` message carrying a
    ``tool_call_response`` part which must NOT be counted (those are
    responses, not invocations).
    """
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[_user_msg('q')])
        _drive_state_through_turns(
            state,
            [
                AssistantMessage(
                    content=[
                        ToolUseBlock(id='t1', name='Bash', input={}),
                        ToolUseBlock(id='t2', name='Read', input={}),
                    ],
                    model='claude-sonnet-4-6',
                ),
                AssistantMessage(
                    content=[
                        ToolUseBlock(id='t3', name='Bash', input={}),
                        TextBlock(text='done'),
                    ],
                    model='claude-sonnet-4-6',
                ),
            ],
        )
        # A tool-response message — must not be counted as an invocation.
        state.add_tool_result(tool_use_id='t3', tool_name='Bash', result='ok')
        state.close()

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    root_attrs = next(s for s in spans if s['name'] == 'invoke_agent')['attributes']
    counts = {entry['tool']: entry['count'] for entry in root_attrs['claude.tools_used']}
    assert counts == {'Bash': 2, 'Read': 1}


@pytest.mark.anyio
async def test_close_skips_tools_used_when_no_tool_calls(exporter: TestExporter) -> None:
    """Text-only conversation → no ``claude.tools_used`` attribute."""
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[_user_msg('hi')])
        _drive_state_through_turns(
            state,
            [AssistantMessage(content=[TextBlock(text='hello')], model='claude-sonnet-4-6')],
        )
        state.close()

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    root_attrs = next(s for s in spans if s['name'] == 'invoke_agent')['attributes']
    assert 'claude.tools_used' not in root_attrs


@pytest.mark.anyio
async def test_close_skips_pydantic_ai_all_messages_when_history_empty(exporter: TestExporter) -> None:
    """Empty ``_history`` (no prompt, no turns) → no ``pydantic_ai.all_messages``
    attribute on the root span."""
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[])
        state.close()

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    root_attrs = next(s for s in spans if s['name'] == 'invoke_agent')['attributes']
    assert 'pydantic_ai.all_messages' not in root_attrs


@pytest.mark.anyio
async def test_handle_assistant_message_populates_per_turn_identifiers(exporter: TestExporter) -> None:
    """All four per-turn identifiers land on the chat span when the
    ``AssistantMessage`` carries them: ``gen_ai.response.id`` (Anthropic
    message id), ``gen_ai.response.finish_reasons`` (per-turn stop reason),
    ``claude.message.uuid`` (SDK stream id), ``claude.parent_tool_use_id``
    (subagent stitching)."""
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[_user_msg('q')])
        state.open_chat_span()
        state.handle_assistant_message(
            AssistantMessage(
                content=[TextBlock(text='done')],
                model='claude-sonnet-4-6',
                message_id='msg_01ABC',
                stop_reason='end_turn',
                uuid='00000000-0000-0000-0000-000000000001',
                parent_tool_use_id='toolu_01PARENT',
            )
        )
        state.close()

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    chat = next(s for s in spans if s['name'].startswith('chat'))
    assert chat['attributes']['gen_ai.response.id'] == 'msg_01ABC'
    assert chat['attributes']['gen_ai.response.finish_reasons'] == ['end_turn']
    assert chat['attributes']['claude.message.uuid'] == '00000000-0000-0000-0000-000000000001'
    assert chat['attributes']['claude.parent_tool_use_id'] == 'toolu_01PARENT'


@pytest.mark.anyio
async def test_handle_assistant_message_skips_missing_identifiers(exporter: TestExporter) -> None:
    """An ``AssistantMessage`` with default (None) optional fields should
    produce no per-turn identifier attributes — avoids emitting noise on
    spans that carry only text content."""
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[_user_msg('q')])
        state.open_chat_span()
        state.handle_assistant_message(AssistantMessage(content=[TextBlock(text='done')], model='claude-sonnet-4-6'))
        state.close()

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    chat = next(s for s in spans if s['name'].startswith('chat'))
    for absent in (
        'gen_ai.response.id',
        'gen_ai.response.finish_reasons',
        'claude.message.uuid',
        'claude.parent_tool_use_id',
    ):
        assert absent not in chat['attributes'], f'{absent!r} should be skipped when None'


@pytest.mark.anyio
async def test_handle_assistant_message_last_write_wins_across_sub_messages(exporter: TestExporter) -> None:
    """Multiple ``AssistantMessage``s on the same chat span → per-turn
    identifiers are last-write-wins (the final ``message_id`` /
    ``stop_reason`` is authoritative for the turn, matching the accumulating
    behaviour of ``OUTPUT_MESSAGES``).

    Locked as a test so a refactor to first-write-wins or list-appending
    is an explicit breaking change, not a silent one.
    """
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[_user_msg('q')])
        state.open_chat_span()
        state.handle_assistant_message(
            AssistantMessage(
                content=[TextBlock(text='thinking...')],
                model='claude-sonnet-4-6',
                message_id='msg_01FIRST',
                stop_reason='tool_use',
            )
        )
        state.handle_assistant_message(
            AssistantMessage(
                content=[TextBlock(text='done')],
                model='claude-sonnet-4-6',
                message_id='msg_01LAST',
                stop_reason='end_turn',
            )
        )
        state.close()

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    chat = next(s for s in spans if s['name'].startswith('chat'))
    assert chat['attributes']['gen_ai.response.id'] == 'msg_01LAST'
    assert chat['attributes']['gen_ai.response.finish_reasons'] == ['end_turn']


def test_options_attrs_full() -> None:
    """``_options_attrs`` surfaces every observability-relevant field on the
    invoke_agent root span when the caller populated it."""
    opts = ClaudeAgentOptions(
        model='claude-sonnet-4-6',
        fallback_model='claude-haiku-4-5',
        permission_mode='acceptEdits',
        max_turns=5,
        max_budget_usd=0.5,
        allowed_tools=['Bash', 'Read'],
        disallowed_tools=['WebFetch'],
        effort='medium',
        resume='prev-session-uuid',
        skills=['skill_a', 'skill_b'],
        setting_sources=['project'],
        continue_conversation=True,
        include_partial_messages=True,
        enable_file_checkpointing=True,
        fork_session=True,
        agents={'reviewer': object(), 'planner': object()},  # type: ignore[dict-item]
    )
    attrs = _options_attrs(opts)
    assert attrs == {
        'claude.options.model': 'claude-sonnet-4-6',
        'claude.options.fallback_model': 'claude-haiku-4-5',
        'claude.permission_mode': 'acceptEdits',
        'claude.max_turns': 5,
        'claude.max_budget_usd': 0.5,
        'claude.effort': 'medium',
        'claude.resume_from': 'prev-session-uuid',
        'claude.skills_mode': 'allowlist',
        'claude.skills': ['skill_a', 'skill_b'],
        'claude.allowed_tools': ['Bash', 'Read'],
        'claude.disallowed_tools': ['WebFetch'],
        'claude.setting_sources': ['project'],
        'claude.continue_conversation': True,
        'claude.include_partial_messages': True,
        'claude.enable_file_checkpointing': True,
        'claude.fork_on_resume': True,
        'claude.agents': ['planner', 'reviewer'],
    }


def test_options_attrs_defaults_emit_nothing() -> None:
    """A default-constructed ``ClaudeAgentOptions()`` yields no extra
    attributes — keeps happy-path spans free of always-empty noise."""
    assert _options_attrs(ClaudeAgentOptions()) == {}


def test_options_attrs_renames_session_substrings() -> None:
    """``resume`` and ``fork_session`` are surfaced under renamed attribute
    keys (``claude.resume_from`` / ``claude.fork_on_resume``) because the
    original names contain the ``session`` substring that triggers the
    default scrubber on attribute names. Locked as a test so a refactor
    that "fixes" the names back to the SDK field names regresses scrubbing.
    """
    opts = ClaudeAgentOptions(resume='r', fork_session=True)
    attrs = _options_attrs(opts)
    assert 'claude.resume_from' in attrs
    assert 'claude.fork_on_resume' in attrs
    assert not any('session' in k for k in attrs)


def test_options_attrs_skills_all_normalises_to_mode_only() -> None:
    """``skills='all'`` emits ``claude.skills_mode='all'`` and NO
    ``claude.skills`` list — keeps the attribute schema stable as a list
    type for downstream typed stores."""
    attrs = _options_attrs(ClaudeAgentOptions(skills='all'))
    assert attrs == {'claude.skills_mode': 'all'}


def test_options_attrs_rejects_non_options_object() -> None:
    """Defensive isinstance — duck-typed mocks emit nothing rather than
    silently shipping garbage attribute values."""
    from types import SimpleNamespace

    fake = SimpleNamespace(
        model=123,  # wrong type
        permission_mode=['nonsense'],  # wrong type
        max_turns='five',  # wrong type
    )
    assert _options_attrs(fake) == {}


def test_options_attrs_skips_non_dict_agents() -> None:
    """``agents`` is typed ``dict[str, AgentDefinition] | None``. If a
    future SDK change makes it a list, our extractor must skip rather
    than emit ``[<AgentDefinition repr…>, …]``."""
    opts = ClaudeAgentOptions()
    object.__setattr__(opts, 'agents', ['reviewer', 'planner'])  # simulate shape drift
    attrs = _options_attrs(opts)
    assert 'claude.agents' not in attrs


def test_server_tool_result_persists_under_assistant_role_in_history() -> None:
    """Server-tool ``tool_call_response`` parts stay under ``role='assistant'``
    in conversation history, not promoted to ``role='tool'``.

    This is intentional: the Anthropic API returns server-executed tool
    results inside the assistant message itself (alongside text and
    server_tool_use blocks), so mirroring that shape preserves the original
    turn structure. Client-side tool results, by contrast, arrive in a
    separate user-role message and get recorded via
    :meth:`_ConversationState.add_tool_result` under ``role='tool'``.

    Locked as a test to prevent a future refactor from silently splitting
    server-tool responses out of the assistant turn.
    """
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[])
        state.open_chat_span()
        asst = SimpleNamespace(
            content=[
                TextBlock(text='Consulting advisor.'),
                ServerToolUseBlock(id='srv_1', name='advisor', input={'q': 'ok?'}),
                ServerToolResultBlock(tool_use_id='srv_1', content={'summary': 'yes'}),
            ],
            model='claude-sonnet-4-6',
            usage=None,
            error=None,
        )
        state.handle_assistant_message(asst)  # type: ignore[arg-type]
        state.close_chat_span()

        # After close, history should carry the assistant turn with both the
        # tool_call and the tool_call_response parts under role='assistant'.
        history = state._history
        assert len(history) == 1
        turn = history[0]
        assert turn['role'] == 'assistant'
        part_kinds = [p.get('type') for p in turn['parts']]
        assert 'tool_call' in part_kinds
        assert 'tool_call_response' in part_kinds


class TestExtractUsage:
    def test_extract_usage(self) -> None:
        usage = {
            'input_tokens': 100,
            'output_tokens': 50,
            'cache_read_input_tokens': 20,
            'cache_creation_input_tokens': 10,
        }
        result = _extract_usage(usage)
        # input_tokens is the total: 100 + 20 + 10 = 130
        assert result['gen_ai.usage.input_tokens'] == 130
        assert result['gen_ai.usage.output_tokens'] == 50
        assert result['gen_ai.usage.cache_read.input_tokens'] == 20
        assert result['gen_ai.usage.cache_creation.input_tokens'] == 10

    def test_extract_empty(self) -> None:
        assert _extract_usage(None) == {}
        assert _extract_usage({}) == {}

    def test_only_cache_read(self) -> None:
        result = _extract_usage({'input_tokens': 50, 'cache_read_input_tokens': 10})
        assert result['gen_ai.usage.input_tokens'] == 60  # total: 50 + 10
        assert result['gen_ai.usage.cache_read.input_tokens'] == 10

    def test_only_cache_create(self) -> None:
        result = _extract_usage({'output_tokens': 30, 'cache_creation_input_tokens': 5})
        assert result['gen_ai.usage.input_tokens'] == 5  # total: 0 + 0 + 5
        assert result['gen_ai.usage.output_tokens'] == 30
        assert result['gen_ai.usage.cache_creation.input_tokens'] == 5

    def test_non_dict_usage(self) -> None:
        class UsageObj:
            input_tokens = 100
            output_tokens = 50
            cache_read_input_tokens = None
            cache_creation_input_tokens = None

        result = _extract_usage(UsageObj())
        assert result == {'gen_ai.usage.input_tokens': 100, 'gen_ai.usage.output_tokens': 50}

    def test_partial_usage(self) -> None:
        result = _extract_usage({'input_tokens': 100, 'output_tokens': 50}, partial=True)
        assert result == {'gen_ai.usage.partial.input_tokens': 100, 'gen_ai.usage.partial.output_tokens': 50}
        assert 'gen_ai.usage.input_tokens' not in result

    def test_invalid_token_values(self) -> None:
        result = _extract_usage({'input_tokens': 'not_a_number', 'output_tokens': None})
        assert result == {}


# ---------------------------------------------------------------------------
# Hook function tests (direct calls, no transport needed).
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_tool_use_hooks(exporter: TestExporter) -> None:
    """Test pre/post tool use hooks create proper child spans."""
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')

    with logfire_instance.span('root') as root_span:
        state = _ConversationState(logfire=logfire_instance, root_span=root_span, input_messages=[])
        _set_state(state)
        try:
            # Successful tool call
            await pre_tool_use_hook(
                {'tool_name': 'Bash', 'tool_input': {'command': 'ls'}, 'tool_use_id': 'tool_1'},
                'tool_1',
                {'signal': None},
            )
            await post_tool_use_hook(
                {
                    'tool_name': 'Bash',
                    'tool_input': {'command': 'ls'},
                    'tool_response': 'file1.txt',
                    'tool_use_id': 'tool_1',
                },
                'tool_1',
                {'signal': None},
            )
            # Successful tool call with no tool_response
            await pre_tool_use_hook(
                {'tool_name': 'Read', 'tool_input': {'path': '/tmp'}, 'tool_use_id': 'tool_no_resp'},
                'tool_no_resp',
                {'signal': None},
            )
            await post_tool_use_hook(
                {'tool_name': 'Read', 'tool_input': {'path': '/tmp'}, 'tool_use_id': 'tool_no_resp'},
                'tool_no_resp',
                {'signal': None},
            )
            # Failed tool call
            await pre_tool_use_hook(
                {'tool_name': 'Write', 'tool_input': {'path': '/tmp/test'}, 'tool_use_id': 'tool_2'},
                'tool_2',
                {'signal': None},
            )
            await post_tool_use_failure_hook(
                {
                    'tool_name': 'Write',
                    'tool_input': {'path': '/tmp/test'},
                    'error': 'Permission denied',
                    'tool_use_id': 'tool_2',
                },
                'tool_2',
                {'signal': None},
            )
        finally:
            _clear_state()

    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'execute_tool Bash',
                'context': {'trace_id': 1, 'span_id': 3, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'start_time': 2000000000,
                'end_time': 3000000000,
                'attributes': {
                    'code.filepath': 'test_claude_agent_sdk.py',
                    'code.function': 'test_tool_use_hooks',
                    'code.lineno': 123,
                    'logfire.msg_template': 'execute_tool Bash',
                    'logfire.msg': 'execute_tool Bash',
                    'gen_ai.operation.name': 'execute_tool',
                    'gen_ai.tool.name': 'Bash',
                    'gen_ai.tool.call.id': 'tool_1',
                    'gen_ai.tool.call.arguments': {'command': 'ls'},
                    'logfire.span_type': 'span',
                    'gen_ai.tool.call.result': 'file1.txt',
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'gen_ai.operation.name': {},
                            'gen_ai.tool.name': {},
                            'gen_ai.tool.call.id': {},
                            'gen_ai.tool.call.arguments': {'type': 'object'},
                            'gen_ai.tool.call.result': {},
                        },
                    },
                },
            },
            {
                'name': 'execute_tool Read',
                'context': {'trace_id': 1, 'span_id': 5, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'start_time': 4000000000,
                'end_time': 5000000000,
                'attributes': {
                    'code.filepath': 'test_claude_agent_sdk.py',
                    'code.function': 'test_tool_use_hooks',
                    'code.lineno': 123,
                    'logfire.msg_template': 'execute_tool Read',
                    'logfire.msg': 'execute_tool Read',
                    'gen_ai.operation.name': 'execute_tool',
                    'gen_ai.tool.name': 'Read',
                    'gen_ai.tool.call.id': 'tool_no_resp',
                    'gen_ai.tool.call.arguments': {'path': '/tmp'},
                    'logfire.span_type': 'span',
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'gen_ai.operation.name': {},
                            'gen_ai.tool.name': {},
                            'gen_ai.tool.call.id': {},
                            'gen_ai.tool.call.arguments': {'type': 'object'},
                        },
                    },
                },
            },
            {
                'name': 'execute_tool Write',
                'context': {'trace_id': 1, 'span_id': 7, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'start_time': 6000000000,
                'end_time': 7000000000,
                'attributes': {
                    'code.filepath': 'test_claude_agent_sdk.py',
                    'code.function': 'test_tool_use_hooks',
                    'code.lineno': 123,
                    'logfire.msg_template': 'execute_tool Write',
                    'logfire.msg': 'execute_tool Write',
                    'gen_ai.operation.name': 'execute_tool',
                    'gen_ai.tool.name': 'Write',
                    'gen_ai.tool.call.id': 'tool_2',
                    'gen_ai.tool.call.arguments': {'path': '/tmp/test'},
                    'logfire.span_type': 'span',
                    'error.type': 'Permission denied',
                    'logfire.level_num': 17,
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'gen_ai.operation.name': {},
                            'gen_ai.tool.name': {},
                            'gen_ai.tool.call.id': {},
                            'gen_ai.tool.call.arguments': {'type': 'object'},
                            'error.type': {},
                        },
                    },
                },
            },
            {
                'name': 'root',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 8000000000,
                'attributes': {
                    'code.filepath': 'test_claude_agent_sdk.py',
                    'code.function': 'test_tool_use_hooks',
                    'code.lineno': 123,
                    'logfire.msg_template': 'root',
                    'logfire.msg': 'root',
                    'logfire.span_type': 'span',
                },
            },
        ]
    )


@pytest.mark.anyio
async def test_clear_orphaned_tool_spans(exporter: TestExporter) -> None:
    """state.close() ends and removes any orphaned tool spans."""
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')

    with logfire_instance.span('root') as root_span:
        state = _ConversationState(logfire=logfire_instance, root_span=root_span, input_messages=[])
        _set_state(state)
        try:
            # Start a tool span but never call post_tool_use_hook
            await pre_tool_use_hook(
                {'tool_name': 'OrphanTool', 'tool_input': {}, 'tool_use_id': 'orphan_1'},
                'orphan_1',
                {'signal': None},
            )
            assert 'orphan_1' in state.active_tool_spans
            state.close()
            assert 'orphan_1' not in state.active_tool_spans
            assert len(state.active_tool_spans) == 0
        finally:
            _clear_state()

    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'execute_tool OrphanTool',
                'context': {'trace_id': 1, 'span_id': 3, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'start_time': 2000000000,
                'end_time': 3000000000,
                'attributes': {
                    'code.filepath': 'test_claude_agent_sdk.py',
                    'code.function': 'test_clear_orphaned_tool_spans',
                    'code.lineno': 123,
                    'logfire.msg_template': 'execute_tool OrphanTool',
                    'logfire.msg': 'execute_tool OrphanTool',
                    'gen_ai.operation.name': 'execute_tool',
                    'gen_ai.tool.name': 'OrphanTool',
                    'gen_ai.tool.call.id': 'orphan_1',
                    'gen_ai.tool.call.arguments': {},
                    'logfire.span_type': 'span',
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'gen_ai.operation.name': {},
                            'gen_ai.tool.name': {},
                            'gen_ai.tool.call.id': {},
                            'gen_ai.tool.call.arguments': {'type': 'object'},
                        },
                    },
                },
            },
            {
                'name': 'root',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 4000000000,
                'attributes': {
                    'code.filepath': 'test_claude_agent_sdk.py',
                    'code.function': 'test_clear_orphaned_tool_spans',
                    'code.lineno': 123,
                    'logfire.msg_template': 'root',
                    'logfire.msg': 'root',
                    'logfire.span_type': 'span',
                },
            },
        ]
    )


@pytest.mark.anyio
async def test_hook_edge_cases() -> None:
    """Hooks return empty dict for edge cases: None tool_use_id, no parent span, missing entry."""
    ctx: HookContext = {'signal': None}

    # None tool_use_id
    assert await pre_tool_use_hook({}, None, ctx) == {}
    assert await post_tool_use_hook({}, None, ctx) == {}
    assert await post_tool_use_failure_hook({}, None, ctx) == {}

    # No state set — hooks bail out
    _clear_state()
    assert await pre_tool_use_hook({'tool_name': 'Bash', 'tool_input': {}}, 'tool_1', ctx) == {}
    assert await post_tool_use_hook({'tool_response': 'test'}, 'nonexistent', ctx) == {}
    assert await post_tool_use_failure_hook({'error': 'test'}, 'nonexistent', ctx) == {}


# ---------------------------------------------------------------------------
# Hook injection tests (use real HookMatcher from SDK).
# ---------------------------------------------------------------------------


def test_inject_hooks_no_hooks_attr() -> None:
    class NoHooksOptions:
        pass

    _inject_tracing_hooks(NoHooksOptions())


def test_inject_hooks_none_hooks() -> None:
    """All 8 hook events register a callback when injection runs.

    Locks the full coverage contract: tool-lifecycle (PreToolUse / PostToolUse
    / PostToolUseFailure) plus the 5 non-tool-lifecycle hooks added in #9
    (UserPromptSubmit / Stop / PreCompact / Notification / PermissionRequest).
    A regression that drops one event silently kills observability for that
    event class — the test is the gate.
    """
    options = ClaudeAgentOptions(hooks=None)
    _inject_tracing_hooks(options)
    assert options.hooks is not None
    expected_events: set[HookEvent] = {
        'PreToolUse',
        'PostToolUse',
        'PostToolUseFailure',
        'UserPromptSubmit',
        'Stop',
        'PreCompact',
        'Notification',
        'PermissionRequest',
        # Issue #3 — closes the last gap, brings SDK-supported hook
        # coverage to 10/10.
        'SubagentStart',
        'SubagentStop',
    }
    assert set(options.hooks) == expected_events
    for event in expected_events:
        assert len(options.hooks[event]) == 1, f'event {event!r} should have exactly one matcher'


def test_inject_hooks_with_existing_events() -> None:
    existing_hook = HookMatcher(matcher='existing', hooks=[pre_tool_use_hook])

    class Opts:
        hooks: dict[str, list[HookMatcher]] | None = {
            'PreToolUse': [existing_hook],
            'PostToolUse': [],
            'PostToolUseFailure': [],
        }

    options = Opts()
    _inject_tracing_hooks(options)
    assert options.hooks is not None
    # Existing user-supplied matcher is preserved AFTER our injected one
    # (we ``insert(0, ...)`` so our hook runs first).
    assert len(options.hooks['PreToolUse']) == 2
    assert options.hooks['PreToolUse'][1] is existing_hook
    # New non-tool-lifecycle events are added too, even though Opts didn't
    # declare them.
    for event in (
        'UserPromptSubmit',
        'Stop',
        'PreCompact',
        'Notification',
        'PermissionRequest',
        'SubagentStart',
        'SubagentStop',
    ):
        assert event in options.hooks
        assert len(options.hooks[event]) == 1


# ---------------------------------------------------------------------------
# Hook-event helper tests (issue #9). Each helper is a small unit that runs
# under an open ``invoke_agent`` span; we drive it directly with synthetic
# input dicts rather than going through the SDK's hook-dispatch machinery.
# Three of the five hooks (PreCompact / Notification / PermissionRequest)
# don't fire under non-interactive SDK transport — these unit tests are the
# only coverage they get, so the assertions are intentionally specific
# (level, attribute names, scrubber-relevance, counter behavior).
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_record_user_prompt_submit_emits_log_with_prompt_attr(exporter: TestExporter) -> None:
    """``_record_user_prompt_submit`` emits an info log under the active
    invoke_agent with ``claude.user_prompt`` carrying the submitted text and
    increments the per-conversation counter.
    """
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[])
        _record_user_prompt_submit(state, {'session_id': 's1', 'prompt': 'hello world'})
    assert state.user_prompt_count == 1

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    log = next(s for s in spans if s['name'] == 'Hook: UserPromptSubmit')
    assert log['attributes']['claude.user_prompt'] == 'hello world'
    assert log['attributes']['gen_ai.conversation.id'] == 's1'
    assert log['attributes']['logfire.level_num'] == 9  # info


@pytest.mark.anyio
async def test_record_user_prompt_submit_skips_when_prompt_absent(exporter: TestExporter) -> None:
    """A defensive run with no ``prompt`` field still increments the counter
    and emits the log (audit value: every hook fires, even malformed ones)
    but skips the ``claude.user_prompt`` attribute."""
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[])
        _record_user_prompt_submit(state, {'session_id': 's1'})
    assert state.user_prompt_count == 1
    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    log = next(s for s in spans if s['name'] == 'Hook: UserPromptSubmit')
    assert 'claude.user_prompt' not in log['attributes']


@pytest.mark.anyio
async def test_record_stop_captures_undeclared_last_assistant_message(exporter: TestExporter) -> None:
    """``last_assistant_message`` is on the wire but absent from the SDK's
    ``StopHookInput`` type. Capturing it via ``.get()`` is the forward-compat
    contract: present today → land it; absent tomorrow → silently skip.
    """
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[])
        _record_stop(
            state,
            {'session_id': 's1', 'stop_hook_active': False, 'last_assistant_message': 'final text'},
        )

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    log = next(s for s in spans if s['name'] == 'Hook: Stop')
    assert log['attributes']['claude.stop.last_assistant_message'] == 'final text'
    # ``stop_hook_active=False`` is the common case — skip it to keep
    # happy-path logs noise-free.
    assert 'claude.stop.hook_active' not in log['attributes']


@pytest.mark.anyio
async def test_record_stop_emits_hook_active_only_when_true(exporter: TestExporter) -> None:
    """Locks the "emit only when True" semantic for ``stop_hook_active``.

    If the SDK ever flips this to default-True the logs would otherwise be
    drowned in always-True attrs; the test is the canary.
    """
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[])
        _record_stop(state, {'session_id': 's1', 'stop_hook_active': True})

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    log = next(s for s in spans if s['name'] == 'Hook: Stop')
    assert log['attributes']['claude.stop.hook_active'] is True


@pytest.mark.anyio
async def test_record_pre_compact_emits_warn_log_with_trigger(exporter: TestExporter) -> None:
    """PreCompact is warn-level (context-pressure signal) and increments
    ``compact_count``. Captures both ``trigger`` and ``custom_instructions``.
    """
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[])
        _record_pre_compact(
            state,
            {'session_id': 's1', 'trigger': 'auto', 'custom_instructions': 'preserve API tokens section'},
        )
    assert state.compact_count == 1

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    log = next(s for s in spans if s['name'].startswith('Hook: PreCompact'))
    assert log['attributes']['claude.compact.trigger'] == 'auto'
    assert log['attributes']['claude.compact.custom_instructions'] == 'preserve API tokens section'
    assert log['attributes']['logfire.level_num'] == 13  # warn


@pytest.mark.anyio
async def test_record_pre_compact_skips_none_custom_instructions(exporter: TestExporter) -> None:
    """``custom_instructions`` is ``str | None`` per the SDK type — None → skip."""
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[])
        _record_pre_compact(
            state,
            {'session_id': 's1', 'trigger': 'manual', 'custom_instructions': None},
        )

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    log = next(s for s in spans if s['name'].startswith('Hook: PreCompact'))
    assert 'claude.compact.custom_instructions' not in log['attributes']
    assert log['attributes']['claude.compact.trigger'] == 'manual'


@pytest.mark.anyio
async def test_record_notification_captures_message_title_type(exporter: TestExporter) -> None:
    """Notification hook surfaces all 3 SDK fields plus the per-conversation counter."""
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[])
        _record_notification(
            state,
            {
                'session_id': 's1',
                'message': 'Permission needed',
                'title': 'Claude Code',
                'notification_type': 'permission_request',
            },
        )
    assert state.notification_count == 1

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    log = next(s for s in spans if s['name'].startswith('Hook: Notification'))
    assert log['attributes']['claude.notification.message'] == 'Permission needed'
    assert log['attributes']['claude.notification.title'] == 'Claude Code'
    assert log['attributes']['claude.notification.type'] == 'permission_request'


@pytest.mark.anyio
async def test_record_permission_request_captures_subagent_attribution(exporter: TestExporter) -> None:
    """PermissionRequest fired from inside a subagent context — capture
    ``agent_id`` / ``agent_type`` opportunistically (groundwork for #3).
    Also locks ``tool_input`` capture under the audit attr.
    """
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[])
        _record_permission_request(
            state,
            {
                'session_id': 's1',
                'tool_name': 'Bash',
                'tool_input': {'command': 'rm -rf tmp/'},
                'permission_suggestions': [{'allow': True}],
                'agent_id': 'subagent-7',
                'agent_type': 'general-purpose',
            },
        )
    assert state.permission_request_count == 1

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    log = next(s for s in spans if s['name'].startswith('Hook: PermissionRequest'))
    assert log['attributes']['gen_ai.tool.name'] == 'Bash'
    assert log['attributes']['claude.permission_request.tool_input'] == {'command': 'rm -rf tmp/'}
    assert log['attributes']['claude.permission_request.suggestions'] == [{'allow': True}]
    assert log['attributes']['claude.agent_id'] == 'subagent-7'
    assert log['attributes']['claude.agent_type'] == 'general-purpose'


@pytest.mark.anyio
async def test_record_permission_request_skips_subagent_attribution_when_absent(exporter: TestExporter) -> None:
    """Main-thread PermissionRequest (no subagent context) → no agent.* attrs."""
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[])
        _record_permission_request(
            state,
            {'session_id': 's1', 'tool_name': 'Read', 'tool_input': {'path': '/tmp/foo'}},
        )
    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    log = next(s for s in spans if s['name'].startswith('Hook: PermissionRequest'))
    for absent in ('claude.agent_id', 'claude.agent_type', 'claude.permission_request.suggestions'):
        assert absent not in log['attributes'], f'{absent!r} should be skipped when not in input'


@pytest.mark.anyio
async def test_close_emits_hook_event_counters_when_nonzero(exporter: TestExporter) -> None:
    """Counters land on the ``invoke_agent`` root span at ``close()`` —
    matches the ``claude.tools_used`` precedent. Regressions that fail to
    set them on the root would silently break dashboard aggregation.
    """
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[])
        # Fire each helper so all four counters increment.
        _record_user_prompt_submit(state, {'session_id': 's1', 'prompt': 'p'})
        _record_pre_compact(state, {'session_id': 's1', 'trigger': 'auto', 'custom_instructions': None})
        _record_notification(state, {'session_id': 's1', 'message': 'm', 'notification_type': 'idle'})
        _record_permission_request(state, {'session_id': 's1', 'tool_name': 'Bash', 'tool_input': {}})
        state.close()

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    root_attrs = next(s for s in spans if s['name'] == 'invoke_agent')['attributes']
    assert root_attrs['claude.user_prompt_count'] == 1
    assert root_attrs['claude.compact_count'] == 1
    assert root_attrs['claude.notification_count'] == 1
    assert root_attrs['claude.permission_request_count'] == 1


@pytest.mark.anyio
async def test_close_skips_zero_hook_event_counters(exporter: TestExporter) -> None:
    """Sessions where no hook events fired don't carry zero-valued counter
    attributes — keeps happy-path spans noise-free.
    """
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[])
        state.close()

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    root_attrs = next(s for s in spans if s['name'] == 'invoke_agent')['attributes']
    for absent in (
        'claude.user_prompt_count',
        'claude.compact_count',
        'claude.notification_count',
        'claude.permission_request_count',
    ):
        assert absent not in root_attrs


@pytest.mark.anyio
async def test_hook_callbacks_no_state_returns_empty() -> None:
    """All five new hook callbacks bail out gracefully when there is no
    active ``_ConversationState`` — covers the case where a hook fires
    outside an instrumented ``receive_response`` block (e.g. SDK initialised
    but no query in flight).
    """
    ctx: HookContext = {'signal': None}
    _clear_state()
    for callback in (
        user_prompt_submit_hook,
        stop_hook,
        pre_compact_hook,
        notification_hook,
        permission_request_hook,
    ):
        assert await callback({'session_id': 's1'}, 'tu_1', ctx) == {}


@pytest.mark.anyio
async def test_hook_callbacks_attach_to_root_span_context(exporter: TestExporter) -> None:
    """A hook callback fires in a different anyio task → contextvars don't
    propagate. Each callback explicitly attaches the root span as the active
    OTel context before emitting, so the resulting log lands as a CHILD of
    ``invoke_agent``, not as an orphan top-level span.

    Locks the integration's most subtle invariant — a regression that drops
    the attach/detach dance produces orphan spans that silently break
    dashboards keyed on root-span filtering.
    """
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    ctx: HookContext = {'signal': None}
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[])
        _set_state(state)
        try:
            await user_prompt_submit_hook({'session_id': 's1', 'prompt': 'hi'}, 'tu_1', ctx)
            await stop_hook({'session_id': 's1', 'stop_hook_active': False}, 'tu_2', ctx)
        finally:
            _clear_state()

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    root_span = next(s for s in spans if s['name'] == 'invoke_agent')
    root_id = root_span['context']['span_id']
    for log_name in ('Hook: UserPromptSubmit', 'Hook: Stop'):
        log = next(s for s in spans if s['name'] == log_name)
        assert log['parent']['span_id'] == root_id, f'{log_name} should be parented to invoke_agent'


# ---------------------------------------------------------------------------
# Permission-flow tests (issue #10).
#
# Three sub-gaps locked in this block: PreToolUse ``updatedInput`` diff,
# ``can_use_tool`` outcome capture, and the deny escalation on the open
# ``execute_tool`` span. Driven through public-API helpers + ``HookContext``
# / ``ToolPermissionContext`` shims so tests don't depend on a live SDK
# control-protocol session.
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_post_tool_use_overwrites_arguments_when_input_mutated(exporter: TestExporter) -> None:
    """A user PreToolUse hook (or ``can_use_tool.updated_input``) that
    mutates ``tool_input`` between pre and execute → ``post_tool_use_hook``
    overwrites ``gen_ai.tool.call.arguments`` with the executed value AND
    surfaces the pre value under ``claude.tool_call.arguments.original``.

    Locks the OTel-canonical "arguments seen by the tool" semantic for
    the canonical attribute, with the diff captured side-by-side.
    """
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    ctx: HookContext = {'signal': None}
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[])
        _set_state(state)
        try:
            # Pre fires with original input.
            await pre_tool_use_hook(
                {'tool_name': 'Bash', 'tool_input': {'command': 'echo hi'}, 'tool_use_id': 'tool_m'},
                'tool_m',
                ctx,
            )
            # Post fires with mutated input — simulates a user PreToolUse hook
            # having returned ``updatedInput`` between us and execution.
            await post_tool_use_hook(
                {
                    'tool_name': 'Bash',
                    'tool_input': {'command': 'echo hi # appended'},
                    'tool_response': 'hi # appended\n',
                    'tool_use_id': 'tool_m',
                },
                'tool_m',
                ctx,
            )
        finally:
            _clear_state()

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    tool_span = next(s for s in spans if s['name'] == 'execute_tool Bash')
    # Canonical attribute now reflects the executed (post) value.
    assert tool_span['attributes']['gen_ai.tool.call.arguments'] == {'command': 'echo hi # appended'}
    # Pre value preserved under the diff attribute.
    assert tool_span['attributes']['claude.tool_call.arguments.original'] == {'command': 'echo hi'}


@pytest.mark.anyio
async def test_post_tool_use_no_extra_attr_when_input_unchanged(exporter: TestExporter) -> None:
    """Happy path: pre and executed inputs match → ``claude.tool_call.arguments.original``
    is NOT emitted. Keeps the trace noise-free for the common case where no
    hook mutated the input.
    """
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    ctx: HookContext = {'signal': None}
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[])
        _set_state(state)
        try:
            await pre_tool_use_hook(
                {'tool_name': 'Read', 'tool_input': {'path': '/tmp/foo'}, 'tool_use_id': 'tool_u'},
                'tool_u',
                ctx,
            )
            await post_tool_use_hook(
                {
                    'tool_name': 'Read',
                    'tool_input': {'path': '/tmp/foo'},
                    'tool_response': 'contents',
                    'tool_use_id': 'tool_u',
                },
                'tool_u',
                ctx,
            )
        finally:
            _clear_state()

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    tool_span = next(s for s in spans if s['name'] == 'execute_tool Read')
    assert 'claude.tool_call.arguments.original' not in tool_span['attributes']
    # Canonical attribute unchanged (pre == executed).
    assert tool_span['attributes']['gen_ai.tool.call.arguments'] == {'path': '/tmp/foo'}


@pytest.mark.anyio
async def test_post_tool_use_failure_clears_original_input_snapshot() -> None:
    """The failure path must also pop ``state.original_tool_inputs`` so
    long-running clients with many failed tool calls don't leak the snapshot
    dict indefinitely.
    """
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    ctx: HookContext = {'signal': None}
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[])
        _set_state(state)
        try:
            await pre_tool_use_hook(
                {'tool_name': 'Write', 'tool_input': {'path': '/x'}, 'tool_use_id': 'tool_f'},
                'tool_f',
                ctx,
            )
            assert 'tool_f' in state.original_tool_inputs
            await post_tool_use_failure_hook(
                {'tool_name': 'Write', 'tool_input': {'path': '/x'}, 'error': 'boom', 'tool_use_id': 'tool_f'},
                'tool_f',
                ctx,
            )
            assert 'tool_f' not in state.original_tool_inputs
        finally:
            _clear_state()


@pytest.mark.anyio
async def test_record_permission_result_allow_with_mutation_logs_updated_input(exporter: TestExporter) -> None:
    """A ``can_use_tool`` Allow that returns a non-None ``updated_input`` /
    ``updated_permissions`` surfaces both on the log. Plain Allow (no
    mutation) is covered separately."""
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[])
        result = PermissionResultAllow(updated_input={'command': 'echo MUTATED'})
        _record_permission_result(state, tool_name='Bash', tool_use_id='t1', result=result)

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    log = next(
        s
        for s in spans
        if s['name'] == 'Hook: PermissionResult ({behavior})'
        and s['attributes'].get('claude.permission_result.behavior') == 'allow'
    )
    assert log['attributes']['claude.permission_result.behavior'] == 'allow'
    assert log['attributes']['gen_ai.tool.name'] == 'Bash'
    assert log['attributes']['claude.permission_result.updated_input'] == {'command': 'echo MUTATED'}
    assert log['attributes']['logfire.level_num'] == 9  # info


@pytest.mark.anyio
async def test_record_permission_result_allow_plain_skips_updated_attrs(exporter: TestExporter) -> None:
    """Plain ``PermissionResultAllow()`` (no mutation) → no
    ``claude.permission_result.updated_input`` / ``updated_permissions``
    attributes. Keeps happy-path Allow logs minimal."""
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[])
        _record_permission_result(state, tool_name='Read', tool_use_id='t2', result=PermissionResultAllow())

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    log = next(
        s
        for s in spans
        if s['name'] == 'Hook: PermissionResult ({behavior})'
        and s['attributes'].get('claude.permission_result.behavior') == 'allow'
    )
    for absent in (
        'claude.permission_result.updated_input',
        'claude.permission_result.updated_permissions',
    ):
        assert absent not in log['attributes']


@pytest.mark.anyio
async def test_record_permission_result_deny_ends_open_execute_tool_span_promptly(exporter: TestExporter) -> None:
    """Locks the most subtle invariant of #10: deny aborts execution before
    PostToolUse fires, leaving an open ``execute_tool`` span. The wrapper
    has to (a) look up that span via ``state.active_tool_spans``, (b) mark
    it ``error.type='PermissionDeny'`` + level=warn + the deny message,
    AND (c) immediately ``pop`` and ``_end`` it.

    The "end promptly" part is the bit that's easy to forget — without it
    the span sits open until ``state.close()`` runs at end-of-conversation,
    producing an end_time that lands at session close and a ``duration_ms``
    that spans the entire remaining session for every denied call. Every
    dashboard / alert keyed off span duration would silently break.

    Asserts: span has both end_time set AND the right attrs, AND no longer
    appears in ``state.active_tool_spans``. Pop also clears the issue-#10
    diff snapshot (no PostToolUse will arrive to consume it).
    """
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[])
        # Mimic ``pre_tool_use_hook`` having opened the execute_tool span
        # AND populated the diff snapshot — both should be cleaned up.
        tool_span = logfire_instance.span('execute_tool WebFetch')
        tool_span._start()
        state.active_tool_spans['t3'] = tool_span
        state.original_tool_inputs['t3'] = {'url': 'https://example.com/'}
        _record_permission_result(
            state,
            tool_name='WebFetch',
            tool_use_id='t3',
            result=PermissionResultDeny(message='blocked by policy', interrupt=False),
        )
        # Helper pops both bookkeeping dicts.
        assert 't3' not in state.active_tool_spans
        assert 't3' not in state.original_tool_inputs
        # Span end was triggered by the helper, not by ``state.close()``.
        # ``_end`` is idempotent so a later ``state.close()`` is harmless.
        assert tool_span._span is not None
        assert tool_span._span.end_time is not None

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    tool = next(s for s in spans if s['name'] == 'execute_tool WebFetch')
    assert tool['attributes']['error.type'] == 'PermissionDeny'
    assert tool['attributes']['claude.permission_result.message'] == 'blocked by policy'
    assert tool['attributes']['logfire.level_num'] == 13  # warn

    log = next(s for s in spans if s['name'].startswith('Hook: PermissionResult'))
    assert log['attributes']['claude.permission_result.behavior'] == 'deny'
    assert log['attributes']['claude.permission_result.message'] == 'blocked by policy'
    # ``interrupt=False`` → attribute skipped (only emit when truthy).
    assert 'claude.permission_result.interrupt' not in log['attributes']
    assert log['attributes']['logfire.level_num'] == 13  # warn


@pytest.mark.anyio
async def test_record_permission_result_deny_with_no_open_span(exporter: TestExporter) -> None:
    """Defensive: if ``can_use_tool`` somehow fires without a matching open
    ``execute_tool`` span (SDK refactors the call order, edge cases with
    parallel tool calls), the deny log still emits cleanly without
    crashing on the missing span lookup."""
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[])
        _record_permission_result(
            state,
            tool_name='WebFetch',
            tool_use_id='nonexistent',
            result=PermissionResultDeny(message='no'),
        )

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    log = next(
        s
        for s in spans
        if s['name'] == 'Hook: PermissionResult ({behavior})'
        and s['attributes'].get('claude.permission_result.behavior') == 'deny'
    )
    assert log['attributes']['claude.permission_result.message'] == 'no'


@pytest.mark.anyio
async def test_record_permission_result_deny_with_interrupt_true(exporter: TestExporter) -> None:
    """``interrupt=True`` is the rare-but-relevant variant — emit the attr
    only in this case so happy-path deny logs stay minimal."""
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[])
        _record_permission_result(
            state,
            tool_name='WebFetch',
            tool_use_id='t4',
            result=PermissionResultDeny(message='abort', interrupt=True),
        )

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    log = next(
        s
        for s in spans
        if s['name'] == 'Hook: PermissionResult ({behavior})'
        and s['attributes'].get('claude.permission_result.behavior') == 'deny'
    )
    assert log['attributes']['claude.permission_result.interrupt'] is True


@pytest.mark.anyio
async def test_wrap_can_use_tool_idempotent_and_passthrough() -> None:
    """``_wrap_can_use_tool`` must:

    - return the same wrapper on a second wrap (the ``_logfire_wrapped``
      sentinel keeps options-object reuse safe across multiple
      ``ClaudeSDKClient`` instances)
    - return whatever the user callback returned, untouched
    - work without an active ``_ConversationState`` (no crash, just no log)
    """
    user_calls: list[tuple[str, dict[str, Any]]] = []

    async def user_callback(tool_name: str, tool_input: dict[str, Any], _ctx: Any) -> Any:
        user_calls.append((tool_name, tool_input))
        return PermissionResultAllow()

    wrapped_once = _wrap_can_use_tool(user_callback)
    wrapped_twice = _wrap_can_use_tool(wrapped_once)
    assert wrapped_once is wrapped_twice

    _clear_state()  # ensure no state — exercises the no-state branch
    ctx = SimpleNamespace(tool_use_id='t1', signal=None, suggestions=[], agent_id=None)
    result = await wrapped_once('Bash', {'command': 'ls'}, ctx)
    assert isinstance(result, PermissionResultAllow)
    assert user_calls == [('Bash', {'command': 'ls'})]


@pytest.mark.anyio
async def test_wrap_can_use_tool_emits_log_under_active_root(exporter: TestExporter) -> None:
    """When a ``_ConversationState`` is active, the wrapper attaches the
    root span as the OTel context before emitting (mirroring the issue-#9
    ``_run_hook`` invariant). The resulting log nests under
    ``invoke_agent``, not as an orphan top-level span.
    """
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')

    async def user_callback(_tool: str, _input: dict[str, Any], _ctx: Any) -> Any:
        return PermissionResultAllow()

    wrapped = _wrap_can_use_tool(user_callback)

    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[])
        _set_state(state)
        try:
            ctx = SimpleNamespace(tool_use_id='t1', signal=None, suggestions=[], agent_id=None)
            await wrapped('Read', {'path': '/x'}, ctx)
        finally:
            _clear_state()

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    root_span = next(s for s in spans if s['name'] == 'invoke_agent')
    log = next(
        s
        for s in spans
        if s['name'] == 'Hook: PermissionResult ({behavior})'
        and s['attributes'].get('claude.permission_result.behavior') == 'allow'
    )
    assert log['parent']['span_id'] == root_span['context']['span_id']


def test_inject_hooks_wraps_can_use_tool_when_set() -> None:
    """``options.can_use_tool`` is opt-in — when set, the integration wraps
    it transparently. Locks the wrap-on-set behavior so a regression that
    skips the wrap silently breaks issue-#10 capture for everyone using
    ``can_use_tool``."""

    async def user_callback(_tool: str, _input: dict[str, Any], _ctx: Any) -> Any:
        return PermissionResultAllow()

    options = ClaudeAgentOptions(can_use_tool=user_callback)
    _inject_tracing_hooks(options)
    assert options.can_use_tool is not user_callback
    assert getattr(options.can_use_tool, '_logfire_wrapped', False) is True


def test_inject_hooks_skips_can_use_tool_when_none() -> None:
    """No callback → no wrap. Don't materialise a wrapper that would force
    the SDK to treat the option as set (it's a feature toggle for the
    streaming-mode permission flow)."""
    options = ClaudeAgentOptions()
    assert options.can_use_tool is None
    _inject_tracing_hooks(options)
    assert options.can_use_tool is None


# ---------------------------------------------------------------------------
# Lifecycle / control-method tests (issue #11).
#
# Each ``ClaudeSDKClient`` lifecycle (``connect`` / ``disconnect``) and
# control method (``set_model`` / ``set_permission_mode`` / ``rewind_files``
# / ``stop_task`` / ``interrupt`` / ``reconnect_mcp_server`` /
# ``toggle_mcp_server``) is wrapped in a span. Six of these don't fire
# under non-interactive SDK transport (need real CLI / MCP / file
# checkpointing setup) and these unit tests are the only coverage they
# get — assertions are intentionally specific.
# ---------------------------------------------------------------------------


def _make_dummy_client() -> ClaudeSDKClient:
    """A ``ClaudeSDKClient`` with a stubbed-in ``_query`` so control methods
    can be invoked without spinning up a real subprocess.

    Distinct from ``_make_client`` (cassette-based, replays a recorded
    session via ``fake_claude.py``): control methods operate on an
    already-connected ``_query`` object — no transport / message stream
    involvement — so a direct stub is the cleanest test surface.

    The integration's ``patched_init`` runs ``_inject_tracing_hooks`` on
    options, which is a no-op for our purposes here.
    """

    class _StubQuery:
        async def interrupt(self) -> None:
            return None

        async def set_permission_mode(self, mode: str) -> None:
            return None

        async def set_model(self, model: str | None) -> None:
            return None

        async def rewind_files(self, user_message_id: str) -> None:
            return None

        async def reconnect_mcp_server(self, server_name: str) -> None:
            return None

        async def toggle_mcp_server(self, server_name: str, enabled: bool) -> None:
            return None

        async def stop_task(self, task_id: str) -> None:
            return None

        async def close(self) -> None:
            return None

    client = ClaudeSDKClient(options=ClaudeAgentOptions())
    # Bypass ``connect`` — populate ``_query`` directly so the control
    # methods' "Not connected" guards pass.
    client._query = _StubQuery()
    client._transport = Mock()
    return client


@pytest.mark.anyio
async def test_receive_messages_produces_full_invoke_agent_tree(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, exporter: TestExporter
) -> None:
    """``receive_messages`` was a complete observability black hole pre-#11.
    The patch makes it produce the same ``invoke_agent`` / ``chat`` /
    ``Hook: *`` tree as ``receive_response``. Locks parity end-to-end via
    a fake-claude cassette replay.
    """
    record = request.config.getoption('--record-claude-cassettes', default=False)
    client = _make_client('basic_conversation.json', monkeypatch=monkeypatch, record=bool(record))
    try:
        await client.connect()
        await client.query('What is 2+2?')
        collected: list[Any] = []
        async for msg in client.receive_messages():
            collected.append(msg)
            if any(type(m).__name__ == 'ResultMessage' for m in collected):
                break
    finally:
        await _close_sdk_streams(client)
        await client.disconnect()

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    names = [s['name'] for s in spans]
    # Single invoke_agent — the nesting fix prevents the inner call
    # to receive_messages from opening a second one when patched.
    assert names.count('invoke_agent') == 1
    # Same body the receive_response cassette test asserts: chat span
    # parented to invoke_agent, lifecycle spans top-level.
    assert any(n.startswith('chat ') for n in names)
    assert 'connect' in names
    assert 'disconnect' in names


@pytest.mark.anyio
async def test_receive_response_calling_receive_messages_does_not_double_invoke_agent(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, exporter: TestExporter
) -> None:
    """``receive_response`` calls ``self.receive_messages()`` internally.
    With both patched, naive wrapping would open two ``invoke_agent`` spans
    (one per wrapper). The ``_drive_invoke_agent`` nesting check at
    ``_get_state() is not None`` prevents this. Locks the contract.
    """
    record = request.config.getoption('--record-claude-cassettes', default=False)
    client = _make_client('basic_conversation.json', monkeypatch=monkeypatch, record=bool(record))
    try:
        await client.connect()
        await client.query('What is 2+2?')
        async for _ in client.receive_response():
            pass
    finally:
        await _close_sdk_streams(client)
        await client.disconnect()

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    invoke_agents = [s for s in spans if s['name'] == 'invoke_agent']
    assert len(invoke_agents) == 1


@pytest.mark.anyio
async def test_connect_error_path_marks_span_with_error_type(exporter: TestExporter) -> None:
    """``CLINotFoundError`` raised during ``connect`` produces a span at
    error level with ``error.type='CLINotFoundError'`` (the short class name
    via ``_annotate_error_span``). Locks the audit-trail for deployments
    where the CLI binary goes missing — pre-#11 these errors vanished into
    user code.
    """
    options = ClaudeAgentOptions(cli_path='/nonexistent/path/claude-cli')
    client = ClaudeSDKClient(options=options)
    with pytest.raises(CLINotFoundError):
        await client.connect()

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    connect_spans = [s for s in spans if s['name'] == 'connect']
    # The SDK's ``connect()`` calls ``self.disconnect()`` in its except
    # cleanup, so a child ``disconnect`` span lives under the failed
    # ``connect``. We're asserting on the outer one.
    failed_connect = next(s for s in connect_spans if s['attributes'].get('error.type') == 'CLINotFoundError')
    assert failed_connect['attributes']['logfire.level_num'] == 17  # error
    assert failed_connect['attributes']['claude.cli_path'] == '/nonexistent/path/claude-cli'


@pytest.mark.anyio
async def test_annotate_error_span_captures_process_error_detail(exporter: TestExporter) -> None:
    """``_annotate_error_span`` uses ``isinstance(exc, ProcessError)`` to
    surface ``exit_code`` + truncated ``stderr``. Other exception types
    only get ``error.type`` + level=error.

    Locked here because a real ``ProcessError`` is hard to provoke in a
    unit test (requires a CLI subprocess that exits non-zero); driving
    the helper directly is the cleanest coverage for the structured-detail
    capture path.
    """
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')

    # 250-char stderr to lock the 200-char truncation.
    long_stderr = 'fatal: ' + ('x' * 250)
    with logfire_instance.span('connect') as span:
        _annotate_error_span(
            span,
            ProcessError(message='subprocess crashed', exit_code=137, stderr=long_stderr),
        )

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    s = next(spans_iter for spans_iter in spans if spans_iter['name'] == 'connect')
    assert s['attributes']['error.type'] == 'ProcessError'
    assert s['attributes']['logfire.level_num'] == 17  # error
    assert s['attributes']['claude.process.exit_code'] == 137
    assert len(s['attributes']['claude.process.stderr']) == 200


@pytest.mark.anyio
async def test_set_model_captures_model_attribute(exporter: TestExporter) -> None:
    """``set_model('claude-sonnet-4-6')`` produces a ``set_model`` span
    with ``gen_ai.request.model`` set — the bit that makes mid-session
    model switches visible for cost attribution.
    """
    client = _make_dummy_client()
    await client.set_model('claude-sonnet-4-6')

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    s = next(s for s in spans if s['name'] == 'set_model')
    assert s['attributes']['gen_ai.request.model'] == 'claude-sonnet-4-6'


@pytest.mark.anyio
async def test_set_model_none_skips_attribute(exporter: TestExporter) -> None:
    """``set_model(None)`` (revert to default) emits the span but no
    ``gen_ai.request.model`` attr — locks the skip-when-None semantic
    so a regression that started emitting ``model='None'`` strings is
    caught."""
    client = _make_dummy_client()
    await client.set_model(None)

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    s = next(s for s in spans if s['name'] == 'set_model')
    assert 'gen_ai.request.model' not in s['attributes']


@pytest.mark.anyio
async def test_set_permission_mode_captures_mode(exporter: TestExporter) -> None:
    """``set_permission_mode('acceptEdits')`` → span with
    ``claude.permission_mode='acceptEdits'``. Reuses the existing
    ``CLAUDE_PERMISSION_MODE`` constant from the options-side capture
    in #8."""
    client = _make_dummy_client()
    await client.set_permission_mode('acceptEdits')

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    s = next(s for s in spans if s['name'] == 'set_permission_mode')
    assert s['attributes']['claude.permission_mode'] == 'acceptEdits'


@pytest.mark.anyio
async def test_rewind_files_captures_user_message_id(exporter: TestExporter) -> None:
    client = _make_dummy_client()
    await client.rewind_files('msg_uuid_abc')

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    s = next(s for s in spans if s['name'] == 'rewind_files')
    assert s['attributes']['claude.rewind.user_message_id'] == 'msg_uuid_abc'


@pytest.mark.anyio
async def test_stop_task_captures_task_id(exporter: TestExporter) -> None:
    client = _make_dummy_client()
    await client.stop_task('task_abc123')

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    s = next(s for s in spans if s['name'] == 'stop_task')
    assert s['attributes']['claude.task_id'] == 'task_abc123'


@pytest.mark.anyio
async def test_interrupt_emits_span_with_no_input_attrs(exporter: TestExporter) -> None:
    """``interrupt`` takes no observable input — the span carries timing
    only. Locks: the patch doesn't accidentally surface unrelated kwargs."""
    client = _make_dummy_client()
    await client.interrupt()

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    s = next(s for s in spans if s['name'] == 'interrupt')
    for absent in (
        'claude.permission_mode',
        'gen_ai.request.model',
        'claude.rewind.user_message_id',
        'claude.task_id',
        'claude.mcp.server_name',
    ):
        assert absent not in s['attributes']


@pytest.mark.anyio
async def test_reconnect_mcp_server_captures_server_name(exporter: TestExporter) -> None:
    client = _make_dummy_client()
    await client.reconnect_mcp_server('my-server')

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    s = next(s for s in spans if s['name'] == 'mcp.reconnect')
    assert s['attributes']['claude.mcp.server_name'] == 'my-server'


@pytest.mark.anyio
async def test_toggle_mcp_server_captures_server_name_and_enabled(exporter: TestExporter) -> None:
    """Multi-arg shape: server_name + enabled both surfaced."""
    client = _make_dummy_client()
    await client.toggle_mcp_server('my-server', enabled=False)

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    s = next(s for s in spans if s['name'] == 'mcp.toggle')
    assert s['attributes']['claude.mcp.server_name'] == 'my-server'
    assert s['attributes']['claude.mcp.enabled'] is False


def test_all_lifecycle_and_control_methods_are_wrapped() -> None:
    """Locks the issue-#11 wrap-every-method contract: every lifecycle and
    control method on ``ClaudeSDKClient`` carries a ``__wrapped__`` sentinel
    pointing at the SDK's original implementation. A regression that drops
    one method's patch (e.g. forgets to add to ``control_method_specs``)
    silently breaks observability for that surface — the test is the gate.

    The autouse fixture has already instrumented at this point.
    """
    cls = ClaudeSDKClient
    for name in (
        'connect',
        'disconnect',
        'receive_messages',
        'interrupt',
        'set_permission_mode',
        'set_model',
        'rewind_files',
        'reconnect_mcp_server',
        'toggle_mcp_server',
        'stop_task',
        # Pre-#11 patches still in place too.
        'query',
        'receive_response',
    ):
        fn = getattr(cls, name)
        assert getattr(fn, '__wrapped__', None) is not None, f'{name} should be wrapped'


# ---------------------------------------------------------------------------
# Subagent / Task* tests (issue #3).
#
# Three layers locked in this block:
# 1. ``SubagentStart`` opens a ``subagent <agent_type>`` span keyed by
#    ``agent_id`` (with optional ``AgentDefinition`` metadata when
#    ``options.agents[agent_type]`` is configured); ``SubagentStop``
#    closes it and captures ``last_assistant_message`` + transcript path.
# 2. ``execute_tool`` spans re-parent under the subagent span when the
#    PreToolUse hook input carries ``agent_id``.
# 3. ``TaskStartedMessage`` / ``TaskProgressMessage`` /
#    ``TaskNotificationMessage`` dispatch as level-appropriate logs.
#
# Plus the parallel-subagent disambiguation case — three concurrent
# subagents with interleaved tool calls all parent correctly via
# ``state.subagent_spans`` keyed by ``agent_id``.
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_record_subagent_start_opens_span_keyed_by_agent_id(exporter: TestExporter) -> None:
    """``SubagentStart`` → opens ``subagent <agent_type>`` span, registers it
    in ``state.subagent_spans[agent_id]``, increments ``subagent_count``.
    """
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[])
        _record_subagent_start(
            state,
            {
                'session_id': 'sess-xyz',
                'agent_id': 'agent_A_id',
                'agent_type': 'echo-helper',
            },
        )
        assert state.subagent_count == 1
        assert 'agent_A_id' in state.subagent_spans
        # Close so the span end_time is set before snapshot inspection.
        _record_subagent_stop(state, {'agent_id': 'agent_A_id', 'agent_type': 'echo-helper'})

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    sub = next(s for s in spans if s['name'] == 'subagent {agent_type}')
    assert sub['attributes']['claude.agent_id'] == 'agent_A_id'
    assert sub['attributes']['claude.agent_type'] == 'echo-helper'
    assert sub['attributes']['gen_ai.agent.name'] == 'echo-helper'
    assert sub['attributes']['gen_ai.conversation.id'] == 'sess-xyz'


@pytest.mark.anyio
async def test_record_subagent_stop_captures_last_assistant_message_and_transcript(exporter: TestExporter) -> None:
    """``SubagentStop`` closes the span and captures the verbatim
    ``last_assistant_message`` (undocumented wire field, mirrors the
    regular ``Stop`` hook from #9) plus ``agent_transcript_path``."""
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[])
        _record_subagent_start(state, {'agent_id': 'agent_X', 'agent_type': 'reviewer'})
        _record_subagent_stop(
            state,
            {
                'agent_id': 'agent_X',
                'agent_type': 'reviewer',
                'last_assistant_message': 'review complete: looks good',
                'agent_transcript_path': '/tmp/.claude/subagents/agent-X.jsonl',
            },
        )
        # Span popped from active dict.
        assert 'agent_X' not in state.subagent_spans

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    sub = next(s for s in spans if s['name'] == 'subagent {agent_type}')
    assert sub['attributes']['claude.subagent.last_assistant_message'] == 'review complete: looks good'
    assert sub['attributes']['claude.agent.transcript_path'] == '/tmp/.claude/subagents/agent-X.jsonl'


@pytest.mark.anyio
async def test_record_subagent_stop_no_open_span_emits_no_crash(exporter: TestExporter) -> None:
    """Defensive: ``SubagentStop`` arriving without a matching ``Start``
    (e.g. SDK bug, message-stream out-of-order) is a no-op rather than a
    crash."""
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[])
        # No matching Start. Must not raise.
        _record_subagent_stop(state, {'agent_id': 'orphan_agent', 'agent_type': 'whatever'})
    # Nothing additional in the export beyond the root invoke_agent.


@pytest.mark.anyio
async def test_subagent_definition_attrs_full_capture() -> None:
    """``_subagent_definition_attrs`` extracts ``AgentDefinition`` metadata
    from ``options.agents[agent_type]`` for the subagent span (issue #3,
    Option B). Mirrors the ``_options_attrs`` pattern from #8 — defensive
    ``isinstance`` guard, optional-field skip, list-when-non-empty."""
    options = ClaudeAgentOptions(
        agents={
            'reviewer': AgentDefinition(
                description='Reviews code changes for safety.',
                prompt='You are a careful reviewer.',
                tools=['Read', 'Grep'],
                disallowedTools=['Bash'],
                model='haiku',
                skills=['code-review'],
                memory='project',
            ),
        },
    )
    attrs = _subagent_definition_attrs(options, 'reviewer')
    assert attrs == {
        'claude.agent.description': 'Reviews code changes for safety.',
        'claude.agent.system_prompt': 'You are a careful reviewer.',
        'claude.agent.model': 'haiku',
        'claude.agent.memory': 'project',
        'claude.agent.tools': ['Read', 'Grep'],
        'claude.agent.disallowed_tools': ['Bash'],
        'claude.agent.skills': ['code-review'],
    }


def test_subagent_definition_attrs_returns_empty_for_missing_or_invalid() -> None:
    """The lookup is best-effort: returns ``{}`` when options is None,
    when ``agents`` isn't a dict, when the agent_type isn't in the map,
    or when the value isn't a recognisable ``AgentDefinition`` (duck-typed
    mocks). Never crashes."""
    # No options.
    assert _subagent_definition_attrs(None, 'reviewer') == {}

    # options.agents is None.
    options_no_agents = ClaudeAgentOptions()
    assert _subagent_definition_attrs(options_no_agents, 'reviewer') == {}

    # options.agents is a dict but agent_type not in it.
    options_with_agents = ClaudeAgentOptions(
        agents={'other': AgentDefinition(description='x', prompt='y')},
    )
    assert _subagent_definition_attrs(options_with_agents, 'reviewer') == {}

    # Defensive: SimpleNamespace mock with the right shape — gets rejected
    # by the isinstance guard so we don't ship garbage attribute values.
    class FakeOpts:
        agents = {'reviewer': SimpleNamespace(model='haiku', description='fake')}

    assert _subagent_definition_attrs(FakeOpts(), 'reviewer') == {}


@pytest.mark.anyio
async def test_record_subagent_start_pulls_agent_definition_when_set(exporter: TestExporter) -> None:
    """End-to-end: ``_record_subagent_start`` looks up
    ``options.agents[agent_type]`` via ``state.options`` and surfaces the
    ``AgentDefinition`` metadata on the subagent span. Without options,
    only the basic identity attrs land."""
    options = ClaudeAgentOptions(
        agents={
            'echo-helper': AgentDefinition(
                description='Trivial echo helper',
                prompt='Echo what you receive',
                tools=['Bash'],
                model='haiku',
            ),
        },
    )
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(
            logfire=logfire_instance,
            root_span=root,
            input_messages=[],
            options=options,
        )
        _record_subagent_start(state, {'agent_id': 'a1', 'agent_type': 'echo-helper'})
        _record_subagent_stop(state, {'agent_id': 'a1', 'agent_type': 'echo-helper'})

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    sub = next(s for s in spans if s['name'] == 'subagent {agent_type}')
    assert sub['attributes']['claude.agent.model'] == 'haiku'
    assert sub['attributes']['claude.agent.description'] == 'Trivial echo helper'
    assert sub['attributes']['claude.agent.system_prompt'] == 'Echo what you receive'
    assert sub['attributes']['claude.agent.tools'] == ['Bash']


@pytest.mark.anyio
async def test_tool_parent_otel_span_picks_subagent_when_agent_id_present(exporter: TestExporter) -> None:
    """Locks the issue-#3 re-parent contract: when ``input_data`` carries
    ``agent_id`` AND ``state.subagent_spans`` has a span for it,
    ``_tool_parent_otel_span`` returns the subagent's OTel span (so the
    next ``execute_tool`` parents under it). Otherwise → root.
    """
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[])
        _record_subagent_start(state, {'agent_id': 'agent_A', 'agent_type': 'helper'})

        # Subagent context — should return subagent's OTel span.
        sub_otel = _tool_parent_otel_span(state, {'agent_id': 'agent_A', 'tool_name': 'Bash'})
        assert sub_otel is not None
        assert sub_otel is state.subagent_spans['agent_A']._span

        # No agent_id — main thread; should return root.
        root_otel = _tool_parent_otel_span(state, {'tool_name': 'Bash'})
        assert root_otel is state.root_span._span

        # agent_id present but no matching subagent span (defensive) → root.
        unknown = _tool_parent_otel_span(state, {'agent_id': 'unknown_agent', 'tool_name': 'Bash'})
        assert unknown is state.root_span._span

        _record_subagent_stop(state, {'agent_id': 'agent_A', 'agent_type': 'helper'})


@pytest.mark.anyio
async def test_pre_tool_use_hook_reparents_under_subagent_when_agent_id_set(exporter: TestExporter) -> None:
    """End-to-end via the public hook callbacks: when ``pre_tool_use_hook``
    fires with ``agent_id`` in the input dict (the SDK marks tool calls
    originating inside a subagent context this way), the resulting
    ``execute_tool`` span has the subagent span as its parent — not the
    root ``invoke_agent``. Locks the most subtle invariant of #3."""
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    ctx: HookContext = {'signal': None}
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[])
        _set_state(state)
        try:
            # Open subagent span first (mimics SubagentStart firing).
            _record_subagent_start(state, {'agent_id': 'agent_A', 'agent_type': 'helper'})

            # Now a tool call from inside the subagent.
            await pre_tool_use_hook(
                {
                    'tool_name': 'Bash',
                    'tool_input': {'command': 'ls'},
                    'tool_use_id': 'tu_sub_1',
                    'agent_id': 'agent_A',
                    'agent_type': 'helper',
                },
                'tu_sub_1',
                ctx,
            )
            await post_tool_use_hook(
                {
                    'tool_name': 'Bash',
                    'tool_input': {'command': 'ls'},
                    'tool_response': 'file1\n',
                    'tool_use_id': 'tu_sub_1',
                    'agent_id': 'agent_A',
                    'agent_type': 'helper',
                },
                'tu_sub_1',
                ctx,
            )

            _record_subagent_stop(state, {'agent_id': 'agent_A', 'agent_type': 'helper'})
        finally:
            _clear_state()

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    tool_span = next(s for s in spans if s['name'] == 'execute_tool Bash')
    sub_span = next(s for s in spans if s['name'] == 'subagent {agent_type}')
    # Tool span's parent IS the subagent span (not the root invoke_agent).
    assert tool_span['parent']['span_id'] == sub_span['context']['span_id']


@pytest.mark.anyio
async def test_pre_tool_use_hook_parents_under_root_when_no_agent_id(exporter: TestExporter) -> None:
    """Symmetric check: a normal (non-subagent) tool call still parents
    under the root ``invoke_agent``. Catches a regression where
    ``_tool_parent_otel_span`` accidentally always selects a subagent
    span (e.g. due to stale state)."""
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    ctx: HookContext = {'signal': None}
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[])
        _set_state(state)
        try:
            # Open and close a subagent span — it should NOT influence
            # subsequent root-thread tool calls.
            _record_subagent_start(state, {'agent_id': 'agent_A', 'agent_type': 'helper'})
            _record_subagent_stop(state, {'agent_id': 'agent_A', 'agent_type': 'helper'})

            # Tool call from the main thread (no agent_id).
            await pre_tool_use_hook(
                {'tool_name': 'Bash', 'tool_input': {'command': 'ls'}, 'tool_use_id': 'tu_root_1'},
                'tu_root_1',
                ctx,
            )
            await post_tool_use_hook(
                {
                    'tool_name': 'Bash',
                    'tool_input': {'command': 'ls'},
                    'tool_response': 'ok',
                    'tool_use_id': 'tu_root_1',
                },
                'tu_root_1',
                ctx,
            )
        finally:
            _clear_state()

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    tool_span = next(s for s in spans if s['name'] == 'execute_tool Bash')
    root_span = next(s for s in spans if s['name'] == 'invoke_agent')
    assert tool_span['parent']['span_id'] == root_span['context']['span_id']


@pytest.mark.anyio
async def test_three_parallel_subagents_each_tool_lands_under_correct_span(exporter: TestExporter) -> None:
    """**The parallel-subagent correctness lock.** Three concurrent
    subagents (A, B, C) with interleaved tool calls. Each subagent's
    tool-lifecycle hook fires with its own ``agent_id`` — the
    ``_tool_parent_otel_span`` lookup keys by agent_id and parents the
    ``execute_tool`` span under the right subagent.

    SubagentStart / SubagentStop ordering simulates real interleaving
    (B starts last but finishes first, etc.) to catch any state-machine
    assumption that subagents complete in-order.
    """
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    ctx: HookContext = {'signal': None}
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[])
        _set_state(state)
        try:
            # Three SubagentStart events (parent dispatched 3 Tasks).
            _record_subagent_start(state, {'agent_id': 'A_id', 'agent_type': 'reviewer'})
            _record_subagent_start(state, {'agent_id': 'B_id', 'agent_type': 'planner'})
            _record_subagent_start(state, {'agent_id': 'C_id', 'agent_type': 'reviewer'})  # same type as A

            # All three subagent spans are open simultaneously.
            assert set(state.subagent_spans) == {'A_id', 'B_id', 'C_id'}

            # Interleaved tool calls — each carries its own agent_id.
            await pre_tool_use_hook(
                {'tool_name': 'Read', 'tool_input': {}, 'tool_use_id': 'tu_A1', 'agent_id': 'A_id'},
                'tu_A1',
                ctx,
            )
            await pre_tool_use_hook(
                {'tool_name': 'Bash', 'tool_input': {}, 'tool_use_id': 'tu_C1', 'agent_id': 'C_id'},
                'tu_C1',
                ctx,
            )
            await pre_tool_use_hook(
                {'tool_name': 'Grep', 'tool_input': {}, 'tool_use_id': 'tu_B1', 'agent_id': 'B_id'},
                'tu_B1',
                ctx,
            )
            # Posts in different order from pre.
            await post_tool_use_hook(
                {
                    'tool_name': 'Bash',
                    'tool_input': {},
                    'tool_response': 'x',
                    'tool_use_id': 'tu_C1',
                    'agent_id': 'C_id',
                },
                'tu_C1',
                ctx,
            )
            await post_tool_use_hook(
                {
                    'tool_name': 'Read',
                    'tool_input': {},
                    'tool_response': 'x',
                    'tool_use_id': 'tu_A1',
                    'agent_id': 'A_id',
                },
                'tu_A1',
                ctx,
            )
            await post_tool_use_hook(
                {
                    'tool_name': 'Grep',
                    'tool_input': {},
                    'tool_response': 'x',
                    'tool_use_id': 'tu_B1',
                    'agent_id': 'B_id',
                },
                'tu_B1',
                ctx,
            )

            # Stops in different order from starts (B finishes first, then A, then C).
            _record_subagent_stop(state, {'agent_id': 'B_id', 'agent_type': 'planner'})
            _record_subagent_stop(state, {'agent_id': 'A_id', 'agent_type': 'reviewer'})
            _record_subagent_stop(state, {'agent_id': 'C_id', 'agent_type': 'reviewer'})

            # All cleanly closed.
            assert state.subagent_spans == {}
            assert state.subagent_count == 3
        finally:
            _clear_state()

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    # Two subagent spans of type 'reviewer' (A and C) and one 'planner' (B).
    sub_spans = [s for s in spans if s['name'] == 'subagent {agent_type}']
    assert len(sub_spans) == 3
    by_agent_id = {s['attributes']['claude.agent_id']: s for s in sub_spans}
    assert set(by_agent_id) == {'A_id', 'B_id', 'C_id'}

    # The disambiguator: each tool span's parent is its subagent's span,
    # NOT a sibling subagent's, NOT the root.
    tool_spans = [s for s in spans if s['name'].startswith('execute_tool ')]
    parent_by_tool_id = {s['attributes']['gen_ai.tool.call.id']: s['parent']['span_id'] for s in tool_spans}
    assert parent_by_tool_id['tu_A1'] == by_agent_id['A_id']['context']['span_id']
    assert parent_by_tool_id['tu_B1'] == by_agent_id['B_id']['context']['span_id']
    assert parent_by_tool_id['tu_C1'] == by_agent_id['C_id']['context']['span_id']


# ---------------------------------------------------------------------------
# Issue #26: chat-span re-parenting under subagent envelope.
#
# These tests use the SimpleNamespace-style direct-call pattern (matching
# the surrounding #3 tests) rather than a cassette: the binding logic is
# pure span-construction and the cassette would only re-prove what the
# unit tests already lock.
# ---------------------------------------------------------------------------


def _task_started_msg(task_id: str, tool_use_id: str) -> Any:
    """Build a TaskStartedMessage-shaped object for ``_record_task_started``.

    ``_record_task_started`` and its helpers use ``getattr(msg, ..., None)``,
    so only the fields the binding logic actually reads need to be set.
    """
    return SimpleNamespace(task_id=task_id, tool_use_id=tool_use_id)


@pytest.mark.anyio
async def test_subagent_chat_spans_nest_under_subagent_envelope(exporter: TestExporter) -> None:
    """**The #26 lock.** A chat span opened via ``handle_user_message`` with
    a ``UserMessage.parent_tool_use_id`` matching a known subagent parents
    under that subagent's span — not the root ``invoke_agent`` — and carries
    the ``claude.in_subagent.*`` annotation attributes for queryability.
    """
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[])
        _set_state(state)
        try:
            # SubagentStart populates state.subagent_spans[agent_id].
            _record_subagent_start(state, {'agent_id': 'agent_X', 'agent_type': 'echo-helper'})
            # TaskStartedMessage populates the parent_tool_use_id → agent_id map.
            _record_task_started(state, _task_started_msg(task_id='agent_X', tool_use_id='tu_parent_1'))

            # Subagent's UserMessage arrives with parent_tool_use_id set.
            msg = UserMessage(content='echo this', parent_tool_use_id='tu_parent_1')
            state.handle_user_message(msg)

            # Close out so spans are exported.
            state.close_chat_span()
            _record_subagent_stop(state, {'agent_id': 'agent_X', 'agent_type': 'echo-helper'})
        finally:
            _clear_state()

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    sub_span = next(s for s in spans if s['name'] == 'subagent {agent_type}')
    chat_span = next(s for s in spans if s['name'] == 'chat')
    assert chat_span['parent']['span_id'] == sub_span['context']['span_id']
    assert chat_span['attributes']['claude.in_subagent.agent_id'] == 'agent_X'
    assert chat_span['attributes']['claude.in_subagent.agent_type'] == 'echo-helper'


@pytest.mark.anyio
async def test_three_parallel_subagents_chat_spans_attribute_correctly(exporter: TestExporter) -> None:
    """Parallel-subagent disambiguation for chat spans. Three subagents
    with overlapping lifetimes; each subagent emits its own chat turns
    interleaved with the others'. Every chat span must land under its
    own subagent — not a sibling subagent's, not the root.
    """
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[])
        _set_state(state)
        try:
            # Three subagents start; bindings populated in arbitrary order.
            _record_subagent_start(state, {'agent_id': 'A_id', 'agent_type': 'reviewer'})
            _record_subagent_start(state, {'agent_id': 'B_id', 'agent_type': 'planner'})
            _record_subagent_start(state, {'agent_id': 'C_id', 'agent_type': 'reviewer'})  # same type as A
            _record_task_started(state, _task_started_msg(task_id='A_id', tool_use_id='tu_A'))
            _record_task_started(state, _task_started_msg(task_id='B_id', tool_use_id='tu_B'))
            _record_task_started(state, _task_started_msg(task_id='C_id', tool_use_id='tu_C'))

            # Interleaved chat opens — each handle_user_message closes the
            # previous chat span (single _current_span) so every open
            # produces a fresh chat span we can inspect.
            state.handle_user_message(UserMessage(content='c1', parent_tool_use_id='tu_A'))
            state.handle_user_message(UserMessage(content='c2', parent_tool_use_id='tu_C'))
            state.handle_user_message(UserMessage(content='c3', parent_tool_use_id='tu_B'))
            state.handle_user_message(UserMessage(content='c4', parent_tool_use_id='tu_A'))
            state.handle_user_message(UserMessage(content='c5', parent_tool_use_id='tu_C'))
            state.close_chat_span()

            _record_subagent_stop(state, {'agent_id': 'B_id', 'agent_type': 'planner'})
            _record_subagent_stop(state, {'agent_id': 'A_id', 'agent_type': 'reviewer'})
            _record_subagent_stop(state, {'agent_id': 'C_id', 'agent_type': 'reviewer'})
        finally:
            _clear_state()

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    sub_spans = [s for s in spans if s['name'] == 'subagent {agent_type}']
    assert len(sub_spans) == 3
    by_agent_id = {s['attributes']['claude.agent_id']: s for s in sub_spans}

    chat_spans = [s for s in spans if s['name'] == 'chat']
    assert len(chat_spans) == 5
    # Each chat span's parent must be the subagent matching its
    # claude.in_subagent.agent_id annotation — not a sibling, not the root.
    for c in chat_spans:
        annotated_agent_id = c['attributes']['claude.in_subagent.agent_id']
        assert c['parent']['span_id'] == by_agent_id[annotated_agent_id]['context']['span_id'], (
            f'chat span annotated with agent_id={annotated_agent_id} parented under '
            f'span_id={c["parent"]["span_id"]}, expected {by_agent_id[annotated_agent_id]["context"]["span_id"]}'
        )


@pytest.mark.anyio
async def test_chat_spans_with_no_active_subagent_attach_to_invoke_agent(exporter: TestExporter) -> None:
    """Regression guard: a UserMessage with no ``parent_tool_use_id`` (the
    common, non-subagent path) keeps parenting chat spans under the root
    ``invoke_agent`` — the #26 patch must not redirect non-subagent traffic.
    """
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[])
        _set_state(state)
        try:
            state.handle_user_message(UserMessage(content='hi'))
            state.close_chat_span()
        finally:
            _clear_state()

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    chat_span = next(s for s in spans if s['name'] == 'chat')
    root_span = next(s for s in spans if s['name'] == 'invoke_agent')
    assert chat_span['parent']['span_id'] == root_span['context']['span_id']
    assert 'claude.in_subagent.agent_id' not in chat_span['attributes']


@pytest.mark.anyio
async def test_chat_span_with_unknown_parent_tool_use_id_falls_back_to_root(exporter: TestExporter) -> None:
    """Defensive degradation: if a UserMessage carries a ``parent_tool_use_id``
    that has no entry in ``parent_tool_use_id_to_agent_id`` (TaskStartedMessage
    didn't arrive yet, or the SDK skipped it), the chat span attaches under
    the root ``invoke_agent`` instead of crashing. No annotation attrs are
    stamped since attribution is unknown.
    """
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[])
        _set_state(state)
        try:
            state.handle_user_message(UserMessage(content='hi', parent_tool_use_id='tu_unknown'))
            state.close_chat_span()
        finally:
            _clear_state()

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    chat_span = next(s for s in spans if s['name'] == 'chat')
    root_span = next(s for s in spans if s['name'] == 'invoke_agent')
    assert chat_span['parent']['span_id'] == root_span['context']['span_id']
    assert 'claude.in_subagent.agent_id' not in chat_span['attributes']


@pytest.mark.anyio
async def test_chat_span_with_stale_parent_tool_use_id_after_stop_falls_back_to_root(exporter: TestExporter) -> None:
    """Distinct from the *unknown-ptui* path: this exercises the
    *cleanup-then-degrade* contract where a binding existed, was dropped
    at ``_record_subagent_stop``, and a late chat message arrives bearing
    the now-stale ``parent_tool_use_id``. Designed degradation per the
    integration's docstring: chat span attaches under root, no annotation.
    Locks the cleanup side-effect that the patch comments call out as
    load-bearing.
    """
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[])
        _set_state(state)
        try:
            _record_subagent_start(state, {'agent_id': 'agent_X', 'agent_type': 'echo-helper'})
            _record_task_started(state, _task_started_msg(task_id='agent_X', tool_use_id='tu_stale'))
            _record_subagent_stop(state, {'agent_id': 'agent_X', 'agent_type': 'echo-helper'})
            # Subagent already stopped — late chat message arrives with
            # the now-stale ptui (rare; would imply SDK out-of-order
            # delivery).
            state.handle_user_message(UserMessage(content='late', parent_tool_use_id='tu_stale'))
            state.close_chat_span()
        finally:
            _clear_state()

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    chat_span = next(s for s in spans if s['name'] == 'chat')
    root_span = next(s for s in spans if s['name'] == 'invoke_agent')
    assert chat_span['parent']['span_id'] == root_span['context']['span_id']
    assert 'claude.in_subagent.agent_id' not in chat_span['attributes']


def test_subagent_stop_clears_parent_tool_use_id_to_agent_id_entries() -> None:
    """``_record_subagent_stop`` drops every ``ptui → agent_id`` entry for
    the closing subagent so the map doesn't grow unboundedly across long
    sessions. Entries for OTHER live subagents must be preserved.
    """
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[])
        _record_subagent_start(state, {'agent_id': 'A_id', 'agent_type': 'reviewer'})
        _record_subagent_start(state, {'agent_id': 'B_id', 'agent_type': 'reviewer'})
        _record_task_started(state, _task_started_msg(task_id='A_id', tool_use_id='tu_A1'))
        _record_task_started(state, _task_started_msg(task_id='A_id', tool_use_id='tu_A2'))
        _record_task_started(state, _task_started_msg(task_id='B_id', tool_use_id='tu_B1'))

        assert state.parent_tool_use_id_to_agent_id == {
            'tu_A1': 'A_id',
            'tu_A2': 'A_id',
            'tu_B1': 'B_id',
        }

        _record_subagent_stop(state, {'agent_id': 'A_id', 'agent_type': 'reviewer'})
        # Both A entries dropped; B preserved.
        assert state.parent_tool_use_id_to_agent_id == {'tu_B1': 'B_id'}

        _record_subagent_stop(state, {'agent_id': 'B_id', 'agent_type': 'reviewer'})
        assert state.parent_tool_use_id_to_agent_id == {}


@pytest.mark.anyio
async def test_state_close_ends_orphan_subagent_spans(exporter: TestExporter) -> None:
    """If a subagent's ``Start`` fires but ``Stop`` never does (session
    abort, error mid-conversation), ``state.close()`` must end the
    orphan span so it doesn't leak open until process exit. Also clears
    ``subagent_spans``."""
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[])
        _record_subagent_start(state, {'agent_id': 'orphan_A', 'agent_type': 'helper'})
        _record_subagent_start(state, {'agent_id': 'orphan_B', 'agent_type': 'helper'})
        # Neither receives a Stop — simulate aborted session.
        state.close()
        assert state.subagent_spans == {}

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    orphans = [s for s in spans if s['name'] == 'subagent {agent_type}']
    assert len(orphans) == 2
    # Both have end_time set (closed by state.close, not still open).
    # ``exporter.exported_spans_as_dict`` only includes finished spans, so
    # presence here is the proof.


@pytest.mark.anyio
async def test_state_close_emits_subagent_count_when_nonzero(exporter: TestExporter) -> None:
    """``claude.subagent_count`` lands on the root span at close, mirroring
    the other counter aggregates from #9. Skipped when zero (happy-path
    sessions don't carry an always-zero attribute)."""
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[])
        _record_subagent_start(state, {'agent_id': 'A', 'agent_type': 'helper'})
        _record_subagent_stop(state, {'agent_id': 'A', 'agent_type': 'helper'})
        _record_subagent_start(state, {'agent_id': 'B', 'agent_type': 'helper'})
        _record_subagent_stop(state, {'agent_id': 'B', 'agent_type': 'helper'})
        state.close()

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    root_attrs = next(s for s in spans if s['name'] == 'invoke_agent')['attributes']
    assert root_attrs['claude.subagent_count'] == 2


@pytest.mark.anyio
async def test_state_close_skips_zero_subagent_count(exporter: TestExporter) -> None:
    """No subagents fired → no ``claude.subagent_count`` on root."""
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[])
        state.close()

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    root_attrs = next(s for s in spans if s['name'] == 'invoke_agent')['attributes']
    assert 'claude.subagent_count' not in root_attrs


@pytest.mark.anyio
async def test_record_task_started_emits_info_log_with_correlation_attrs(exporter: TestExporter) -> None:
    """``TaskStartedMessage`` → info log with ``claude.task_id``,
    ``claude.task.description``, ``claude.task.type``, and the parent's
    ``gen_ai.tool.call.id`` (the Task tool's id) for cross-span correlation."""
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[])
        msg = SimpleNamespace(
            task_id='task_001',
            description='Echo a phrase via Bash',
            uuid='uuid-task-1',
            session_id='sess-x',
            tool_use_id='toolu_parent_call',
            task_type='local_agent',
        )
        _record_task_started(state, msg)

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    log = next(s for s in spans if s['name'] == 'Task started')
    assert log['attributes']['claude.task_id'] == 'task_001'
    assert log['attributes']['claude.task.description'] == 'Echo a phrase via Bash'
    assert log['attributes']['claude.task.type'] == 'local_agent'
    assert log['attributes']['gen_ai.tool.call.id'] == 'toolu_parent_call'
    assert log['attributes']['gen_ai.conversation.id'] == 'sess-x'
    assert log['attributes']['logfire.level_num'] == 9  # info


@pytest.mark.anyio
async def test_record_task_progress_emits_info_log_with_usage(exporter: TestExporter) -> None:
    """``TaskProgressMessage`` → info log with ``claude.task.usage`` (the
    full TaskUsage dict — total_tokens, tool_uses, duration_ms) and
    ``claude.task.last_tool_name``."""
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[])
        msg = SimpleNamespace(
            task_id='task_002',
            description='Long-running analysis',
            usage={'total_tokens': 5000, 'tool_uses': 3, 'duration_ms': 12000},
            uuid='uuid-task-2',
            session_id='sess-y',
            tool_use_id='toolu_parent_call_2',
            last_tool_name='Bash',
        )
        _record_task_progress(state, msg)

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    log = next(s for s in spans if s['name'] == 'Task progress')
    assert log['attributes']['claude.task_id'] == 'task_002'
    assert log['attributes']['claude.task.last_tool_name'] == 'Bash'
    assert log['attributes']['claude.task.usage'] == {
        'total_tokens': 5000,
        'tool_uses': 3,
        'duration_ms': 12000,
    }
    assert log['attributes']['logfire.level_num'] == 9  # info


@pytest.mark.anyio
async def test_record_task_notification_status_completed_is_info(exporter: TestExporter) -> None:
    """``status='completed'`` → info level."""
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[])
        msg = SimpleNamespace(
            task_id='task_ok',
            status='completed',
            output_file='/tmp/task_ok.txt',
            summary='Done',
            uuid='u',
            session_id='s',
            tool_use_id='t',
            usage=None,
        )
        _record_task_notification(state, msg)

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    log = next(s for s in spans if s['name'] == 'Task {status}')
    assert log['attributes']['claude.task.status'] == 'completed'
    assert log['attributes']['claude.task.summary'] == 'Done'
    assert log['attributes']['claude.task.output_file'] == '/tmp/task_ok.txt'
    assert log['attributes']['logfire.level_num'] == 9  # info


@pytest.mark.anyio
async def test_record_task_notification_status_failed_is_error(exporter: TestExporter) -> None:
    """``status='failed'`` → error level. Audit-relevant: subagent crashes
    surface in error dashboards even when the parent recovers gracefully."""
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[])
        msg = SimpleNamespace(
            task_id='task_fail',
            status='failed',
            output_file='',
            summary='subagent hit an internal error',
            uuid='u',
            session_id='s',
            tool_use_id='t',
            usage=None,
        )
        _record_task_notification(state, msg)

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    log = next(s for s in spans if s['name'] == 'Task {status}')
    assert log['attributes']['logfire.level_num'] == 17  # error


@pytest.mark.anyio
async def test_record_task_notification_status_stopped_is_warn(exporter: TestExporter) -> None:
    """``status='stopped'`` → warn level. User-initiated cancellations
    (via ``stop_task`` from #11) shouldn't be filed as errors but ARE
    audit-relevant."""
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent') as root:
        state = _ConversationState(logfire=logfire_instance, root_span=root, input_messages=[])
        msg = SimpleNamespace(
            task_id='task_stop',
            status='stopped',
            output_file='',
            summary='stopped by user',
            uuid='u',
            session_id='s',
            tool_use_id='t',
            usage=None,
        )
        _record_task_notification(state, msg)

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    log = next(s for s in spans if s['name'] == 'Task {status}')
    assert log['attributes']['logfire.level_num'] == 13  # warn


@pytest.mark.anyio
async def test_already_instrumented() -> None:
    """Calling instrument twice is a no-op (idempotent)."""
    logfire.instrument_claude_agent_sdk()
    logfire.instrument_claude_agent_sdk()
    # No error, and only one layer of patching


# ---------------------------------------------------------------------------
# Cassette-based integration tests (replay through real transport).
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_basic_conversation_cassette(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, exporter: TestExporter
) -> None:
    """Basic conversation replayed from cassette produces correct spans."""
    record = request.config.getoption('--record-claude-cassettes', default=False)
    client = _make_client('basic_conversation.json', monkeypatch=monkeypatch, record=bool(record))
    try:
        await client.connect()
        await client.query('What is 2+2?')
        collected = [msg async for msg in client.receive_response()]
    finally:
        await _close_sdk_streams(client)
        await client.disconnect()

    assert len(collected) >= 2  # system + assistant + result

    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'connect',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'code.filepath': 'test_claude_agent_sdk.py',
                    'code.function': 'test_basic_conversation_cassette',
                    'code.lineno': 123,
                    'claude.cli_path': IsStr(regex=r'.*/fake_claude\.py$'),
                    'logfire.msg_template': 'connect',
                    'logfire.msg': 'connect',
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {'claude.cli_path': {}},
                    },
                    'logfire.span_type': 'span',
                },
            },
            {
                'name': 'chat claude-sonnet-4-6',
                'context': {'trace_id': 2, 'span_id': 5, 'is_remote': False},
                'parent': {'trace_id': 2, 'span_id': 3, 'is_remote': False},
                'start_time': 4000000000,
                'end_time': 5000000000,
                'attributes': {
                    'code.filepath': 'test_claude_agent_sdk.py',
                    'code.function': 'test_basic_conversation_cassette',
                    'code.lineno': 123,
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.provider.name': 'anthropic',
                    'gen_ai.system': 'anthropic',
                    'gen_ai.input.messages': [{'role': 'user', 'parts': [{'type': 'text', 'content': 'What is 2+2?'}]}],
                    'gen_ai.system_instructions': [{'type': 'text', 'content': 'Be helpful'}],
                    'logfire.msg_template': 'chat',
                    'gen_ai.output.messages': [{'role': 'assistant', 'parts': [{'type': 'text', 'content': '4'}]}],
                    'logfire.msg': 'chat claude-sonnet-4-6',
                    'logfire.span_type': 'span',
                    'gen_ai.usage.partial.input_tokens': 9344,
                    'gen_ai.usage.partial.output_tokens': 1,
                    'gen_ai.usage.partial.cache_read.input_tokens': 7166,
                    'gen_ai.usage.partial.cache_creation.input_tokens': 2175,
                    'gen_ai.response.id': 'msg_01BxK8UH2LyuFLVXSPamqHam',
                    'claude.message.uuid': 'ce1101bc-27c6-41b1-ae16-5edf06b0927c',
                    'gen_ai.request.model': 'claude-sonnet-4-6',
                    'gen_ai.response.model': 'claude-sonnet-4-6',
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'gen_ai.operation.name': {},
                            'gen_ai.provider.name': {},
                            'gen_ai.system': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'gen_ai.output.messages': {'type': 'array'},
                            'gen_ai.system_instructions': {'type': 'array'},
                            'gen_ai.request.model': {},
                            'gen_ai.response.model': {},
                            'gen_ai.usage.partial.input_tokens': {},
                            'gen_ai.usage.partial.output_tokens': {},
                            'gen_ai.usage.partial.cache_read.input_tokens': {},
                            'gen_ai.usage.partial.cache_creation.input_tokens': {},
                            'gen_ai.response.id': {},
                            'claude.message.uuid': {},
                        },
                    },
                },
            },
            {
                'name': 'invoke_agent',
                'context': {'trace_id': 2, 'span_id': 3, 'is_remote': False},
                'parent': None,
                'start_time': 3000000000,
                'end_time': 6000000000,
                'attributes': {
                    'code.filepath': 'test_claude_agent_sdk.py',
                    'code.function': 'test_basic_conversation_cassette',
                    'code.lineno': 123,
                    'gen_ai.operation.name': 'invoke_agent',
                    'gen_ai.provider.name': 'anthropic',
                    'gen_ai.system': 'anthropic',
                    'gen_ai.agent.name': 'claude-code',
                    'gen_ai.input.messages': [{'role': 'user', 'parts': [{'type': 'text', 'content': 'What is 2+2?'}]}],
                    'gen_ai.system_instructions': [{'type': 'text', 'content': 'Be helpful'}],
                    'logfire.msg_template': 'invoke_agent',
                    'logfire.msg': 'invoke_agent',
                    'logfire.span_type': 'span',
                    'gen_ai.usage.input_tokens': 9344,
                    'gen_ai.usage.output_tokens': 5,
                    'gen_ai.usage.cache_read.input_tokens': 7166,
                    'gen_ai.usage.cache_creation.input_tokens': 2175,
                    'operation.cost': 0.01039005,
                    'gen_ai.conversation.id': '7ed3c21d-374b-491a-8c66-05e191f6a0be',
                    'num_turns': 1,
                    'duration_ms': 2263,
                    'duration_api_ms': 2257,
                    'claude.result.subtype': 'success',
                    'claude.result.text': '4',
                    'gen_ai.response.finish_reasons': ['end_turn'],
                    'claude.model_usage': {
                        'claude-sonnet-4-6': {
                            'inputTokens': 3,
                            'outputTokens': 5,
                            'cacheReadInputTokens': 7166,
                            'cacheCreationInputTokens': 2175,
                            'webSearchRequests': 0,
                            'costUSD': 0.01039005,
                            'contextWindow': 200000,
                            'maxOutputTokens': 32000,
                        }
                    },
                    'gen_ai.request.model': 'claude-sonnet-4-6',
                    'gen_ai.response.model': 'claude-sonnet-4-6',
                    'pydantic_ai.all_messages': [
                        {'role': 'user', 'parts': [{'type': 'text', 'content': 'What is 2+2?'}]},
                        {'role': 'assistant', 'parts': [{'type': 'text', 'content': '4'}]},
                    ],
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'gen_ai.operation.name': {},
                            'gen_ai.provider.name': {},
                            'gen_ai.system': {},
                            'gen_ai.agent.name': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'gen_ai.system_instructions': {'type': 'array'},
                            'gen_ai.usage.input_tokens': {},
                            'gen_ai.usage.output_tokens': {},
                            'gen_ai.usage.cache_read.input_tokens': {},
                            'gen_ai.usage.cache_creation.input_tokens': {},
                            'operation.cost': {},
                            'gen_ai.conversation.id': {},
                            'num_turns': {},
                            'duration_ms': {},
                            'duration_api_ms': {},
                            'claude.result.subtype': {},
                            'claude.result.text': {},
                            'gen_ai.response.finish_reasons': {'type': 'array'},
                            'claude.model_usage': {'type': 'object'},
                            'gen_ai.request.model': {},
                            'gen_ai.response.model': {},
                            'pydantic_ai.all_messages': {'type': 'array'},
                        },
                    },
                },
            },
            {
                'name': 'disconnect',
                'context': {'trace_id': 3, 'span_id': 7, 'is_remote': False},
                'parent': None,
                'start_time': 7000000000,
                'end_time': 8000000000,
                'attributes': {
                    'code.filepath': 'test_claude_agent_sdk.py',
                    'code.function': 'test_basic_conversation_cassette',
                    'code.lineno': 123,
                    'logfire.msg_template': 'disconnect',
                    'logfire.span_type': 'span',
                    'logfire.msg': 'disconnect',
                },
            },
        ]
    )


@pytest.mark.anyio
async def test_tool_use_conversation_cassette(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, exporter: TestExporter
) -> None:
    """Tool use conversation: assistant calls Bash, gets result, then responds."""
    record = request.config.getoption('--record-claude-cassettes', default=False)
    client = _make_client('tool_use_conversation.json', monkeypatch=monkeypatch, record=bool(record))
    try:
        await client.connect()
        await client.query('List files in the current directory')
        collected = [msg async for msg in client.receive_response()]
    finally:
        await _close_sdk_streams(client)
        await client.disconnect()

    # assistant (tool_use) + user (tool_result) + assistant (text) + result
    assert len(collected) >= 3

    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'chat claude-sonnet-4-6',
                'context': {'trace_id': 1, 'span_id': 3, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'start_time': 2000000000,
                'end_time': 3000000000,
                'attributes': {
                    'code.filepath': IsStr(),
                    'code.function': 'test_tool_use_conversation_cassette',
                    'code.lineno': 123,
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.provider.name': 'anthropic',
                    'gen_ai.system': 'anthropic',
                    'gen_ai.input.messages': [
                        {'role': 'user', 'parts': [{'type': 'text', 'content': 'List files in the current directory'}]}
                    ],
                    'gen_ai.system_instructions': [{'type': 'text', 'content': 'Be helpful'}],
                    'gen_ai.output.messages': [
                        {
                            'role': 'assistant',
                            'parts': [
                                {'type': 'reasoning', 'content': 'Let me list the files in the current directory.'},
                                {
                                    'type': 'tool_call',
                                    'id': 'toolu_01MRdgcFhYNo1LHvRQKvKckg',
                                    'name': 'Bash',
                                    'arguments': {'command': 'ls', 'description': 'List files in current directory'},
                                },
                            ],
                        }
                    ],
                    'gen_ai.response.model': 'claude-sonnet-4-6',
                    'gen_ai.usage.partial.input_tokens': 9343,
                    'gen_ai.request.model': 'claude-sonnet-4-6',
                    'gen_ai.usage.partial.output_tokens': 0,
                    'gen_ai.usage.partial.cache_read.input_tokens': 8313,
                    'gen_ai.usage.partial.cache_creation.input_tokens': 1027,
                    'logfire.msg_template': 'chat',
                    'logfire.msg': 'chat claude-sonnet-4-6',
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'gen_ai.operation.name': {},
                            'gen_ai.provider.name': {},
                            'gen_ai.system': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'gen_ai.system_instructions': {'type': 'array'},
                            'gen_ai.output.messages': {'type': 'array'},
                            'gen_ai.request.model': {},
                            'gen_ai.response.model': {},
                            'gen_ai.usage.partial.input_tokens': {},
                            'gen_ai.usage.partial.output_tokens': {},
                            'gen_ai.usage.partial.cache_read.input_tokens': {},
                            'gen_ai.usage.partial.cache_creation.input_tokens': {},
                        },
                    },
                    'logfire.span_type': 'span',
                },
            },
            {
                'name': 'execute_tool Bash',
                'context': {'trace_id': 1, 'span_id': 5, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'start_time': 4000000000,
                'end_time': 5000000000,
                'attributes': {
                    'code.filepath': IsStr(),
                    'code.lineno': 123,
                    'logfire.msg_template': 'execute_tool Bash',
                    'gen_ai.tool.name': 'Bash',
                    'gen_ai.tool.call.id': 'toolu_01MRdgcFhYNo1LHvRQKvKckg',
                    'gen_ai.tool.call.arguments': {'command': 'ls', 'description': 'List files in current directory'},
                    'gen_ai.tool.call.result': {
                        'stdout': """\
CHANGELOG.md
CLAUDE.md
CONTRIBUTING.md
LICENSE
Makefile
README.md
dist
docs
examples
ignoreme
logfire
logfire-api
mkdocs.yml
plans
pyodide_test
pyproject.toml
release
scratch
site
specs
tests
uv.lock\
""",
                        'stderr': '',
                        'interrupted': False,
                        'isImage': False,
                        'noOutputExpected': False,
                    },
                    'logfire.msg': 'execute_tool Bash',
                    'gen_ai.operation.name': 'execute_tool',
                    'logfire.span_type': 'span',
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'gen_ai.operation.name': {},
                            'gen_ai.tool.name': {},
                            'gen_ai.tool.call.id': {},
                            'gen_ai.tool.call.arguments': {'type': 'object'},
                            'gen_ai.tool.call.result': {'type': 'object'},
                        },
                    },
                },
            },
            {
                'name': 'chat claude-sonnet-4-6',
                'context': {'trace_id': 1, 'span_id': 7, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'start_time': 6000000000,
                'end_time': 7000000000,
                'attributes': {
                    'code.filepath': 'test_claude_agent_sdk.py',
                    'code.function': 'test_tool_use_conversation_cassette',
                    'code.lineno': 123,
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.provider.name': 'anthropic',
                    'gen_ai.system': 'anthropic',
                    'gen_ai.response.model': 'claude-sonnet-4-6',
                    'gen_ai.system_instructions': [{'type': 'text', 'content': 'Be helpful'}],
                    'gen_ai.output.messages': [
                        {
                            'role': 'assistant',
                            'parts': [
                                {
                                    'type': 'text',
                                    'content': """\
Here are the files and directories in the current directory:

| Name | Type |
|------|------|
| `CHANGELOG.md` | File |
| `CLAUDE.md` | File |
| `CONTRIBUTING.md` | File |
| `LICENSE` | File |
| `Makefile` | File |
| `README.md` | File |
| `mkdocs.yml` | File |
| `pyproject.toml` | File |
| `uv.lock` | File |
| `dist/` | Directory |
| `docs/` | Directory |
| `examples/` | Directory |
| `ignoreme/` | Directory |
| `logfire/` | Directory |
| `logfire-api/` | Directory |
| `plans/` | Directory |
| `pyodide_test/` | Directory |
| `release/` | Directory |
| `scratch/` | Directory |
| `site/` | Directory |
| `specs/` | Directory |
| `tests/` | Directory |

There are **9 files** and **12 directories** in the current directory. It looks like a Python project (given `pyproject.toml`, `uv.lock`) — likely the **Logfire** SDK or library based on the `logfire/` and `logfire-api/` directories.\
""",
                                }
                            ],
                        }
                    ],
                    'gen_ai.input.messages': [
                        {
                            'role': 'user',
                            'parts': [{'type': 'text', 'content': 'List files in the current directory'}],
                        },
                        {
                            'role': 'assistant',
                            'parts': [
                                {'type': 'reasoning', 'content': 'Let me list the files in the current directory.'},
                                {
                                    'type': 'tool_call',
                                    'id': 'toolu_01MRdgcFhYNo1LHvRQKvKckg',
                                    'name': 'Bash',
                                    'arguments': {'command': 'ls', 'description': 'List files in current directory'},
                                },
                            ],
                        },
                        {
                            'role': 'tool',
                            'parts': [
                                {
                                    'type': 'tool_call_response',
                                    'id': 'toolu_01MRdgcFhYNo1LHvRQKvKckg',
                                    'name': 'Bash',
                                    'response': {
                                        'stdout': """\
CHANGELOG.md
CLAUDE.md
CONTRIBUTING.md
LICENSE
Makefile
README.md
dist
docs
examples
ignoreme
logfire
logfire-api
mkdocs.yml
plans
pyodide_test
pyproject.toml
release
scratch
site
specs
tests
uv.lock\
""",
                                        'stderr': '',
                                        'interrupted': False,
                                        'isImage': False,
                                        'noOutputExpected': False,
                                    },
                                }
                            ],
                        },
                    ],
                    'gen_ai.usage.partial.input_tokens': 9529,
                    'gen_ai.request.model': 'claude-sonnet-4-6',
                    'gen_ai.usage.partial.output_tokens': 1,
                    'gen_ai.usage.partial.cache_read.input_tokens': 9340,
                    'gen_ai.usage.partial.cache_creation.input_tokens': 188,
                    'logfire.msg_template': 'chat',
                    'logfire.msg': 'chat claude-sonnet-4-6',
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'gen_ai.operation.name': {},
                            'gen_ai.provider.name': {},
                            'gen_ai.system': {},
                            'gen_ai.response.model': {},
                            'gen_ai.system_instructions': {'type': 'array'},
                            'gen_ai.output.messages': {'type': 'array'},
                            'gen_ai.request.model': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'gen_ai.usage.partial.input_tokens': {},
                            'gen_ai.usage.partial.output_tokens': {},
                            'gen_ai.usage.partial.cache_read.input_tokens': {},
                            'gen_ai.usage.partial.cache_creation.input_tokens': {},
                        },
                    },
                    'logfire.span_type': 'span',
                },
            },
            {
                'name': 'invoke_agent',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 8000000000,
                'attributes': {
                    'code.filepath': 'test_claude_agent_sdk.py',
                    'code.function': 'test_tool_use_conversation_cassette',
                    'code.lineno': 123,
                    'gen_ai.operation.name': 'invoke_agent',
                    'gen_ai.provider.name': 'anthropic',
                    'gen_ai.system': 'anthropic',
                    'gen_ai.system_instructions': [{'type': 'text', 'content': 'Be helpful'}],
                    'gen_ai.input.messages': [
                        {
                            'role': 'user',
                            'parts': [{'type': 'text', 'content': 'List files in the current directory'}],
                        }
                    ],
                    'gen_ai.usage.input_tokens': 18872,
                    'gen_ai.usage.output_tokens': 415,
                    'gen_ai.usage.cache_read.input_tokens': 17653,
                    'gen_ai.usage.cache_creation.input_tokens': 1215,
                    'logfire.msg_template': 'invoke_agent',
                    'operation.cost': 0.01608915,
                    'gen_ai.conversation.id': 'ca03765b-a7e1-483b-9629-448c7aba5e7a',
                    'num_turns': 2,
                    'duration_ms': 9352,
                    'logfire.msg': 'invoke_agent',
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'gen_ai.operation.name': {},
                            'gen_ai.provider.name': {},
                            'gen_ai.system': {},
                            'gen_ai.system_instructions': {'type': 'array'},
                            'gen_ai.input.messages': {'type': 'array'},
                            'gen_ai.usage.input_tokens': {},
                            'gen_ai.usage.output_tokens': {},
                            'gen_ai.usage.cache_read.input_tokens': {},
                            'gen_ai.usage.cache_creation.input_tokens': {},
                            'operation.cost': {},
                            'gen_ai.conversation.id': {},
                            'num_turns': {},
                            'duration_ms': {},
                            'gen_ai.request.model': {},
                            'gen_ai.response.model': {},
                        },
                    },
                    'gen_ai.request.model': 'claude-sonnet-4-6',
                    'gen_ai.response.model': 'claude-sonnet-4-6',
                    'logfire.span_type': 'span',
                },
            },
        ]
    )


@pytest.mark.anyio
async def test_server_tool_blocks_cassette(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, exporter: TestExporter
) -> None:
    """ServerToolUseBlock / ServerToolResultBlock in an assistant message
    are emitted as properly-typed ``tool_call`` / ``tool_call_response``
    parts under ``gen_ai.output.messages``.

    Cassette is ``basic_conversation.json`` with the assistant entry
    replaced by one that contains text + server_tool_use + advisor_tool_result
    + text.
    """
    record = request.config.getoption('--record-claude-cassettes', default=False)
    # Recording isn't meaningful here — the synthetic assistant content is
    # model-driven and varies run-to-run. Skip so CI never overwrites it.
    if record:  # pragma: no cover
        pytest.skip('server_tool_conversation.json is hand-assembled, not recorded')

    client = _make_client('server_tool_conversation.json', monkeypatch=monkeypatch)
    try:
        await client.connect()
        await client.query('What is 2+2?')
        collected = [msg async for msg in client.receive_response()]
    finally:
        await _close_sdk_streams(client)
        await client.disconnect()

    # SDK should have parsed the server tool blocks into typed dataclasses.
    asst = next(m for m in collected if isinstance(m, AssistantMessage))
    assert any(isinstance(b, ServerToolUseBlock) for b in asst.content), 'SDK did not yield ServerToolUseBlock'
    assert any(isinstance(b, ServerToolResultBlock) for b in asst.content), 'SDK did not yield ServerToolResultBlock'

    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'connect',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'code.filepath': 'test_claude_agent_sdk.py',
                    'code.function': 'test_server_tool_blocks_cassette',
                    'code.lineno': 123,
                    'claude.cli_path': IsStr(regex=r'.*/fake_claude\.py$'),
                    'logfire.msg_template': 'connect',
                    'logfire.span_type': 'span',
                    'logfire.msg': 'connect',
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {'claude.cli_path': {}},
                    },
                },
            },
            {
                'name': 'chat claude-sonnet-4-6',
                'context': {'trace_id': 2, 'span_id': 5, 'is_remote': False},
                'parent': {'trace_id': 2, 'span_id': 3, 'is_remote': False},
                'start_time': 4000000000,
                'end_time': 5000000000,
                'attributes': {
                    'code.filepath': 'test_claude_agent_sdk.py',
                    'code.function': 'test_server_tool_blocks_cassette',
                    'code.lineno': 123,
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.provider.name': 'anthropic',
                    'gen_ai.system': 'anthropic',
                    'gen_ai.input.messages': [{'role': 'user', 'parts': [{'type': 'text', 'content': 'What is 2+2?'}]}],
                    'gen_ai.system_instructions': [{'type': 'text', 'content': 'Be helpful'}],
                    'logfire.msg_template': 'chat',
                    'gen_ai.output.messages': [
                        {
                            'role': 'assistant',
                            'parts': [
                                {'type': 'text', 'content': 'Checking with advisor.'},
                                {
                                    'type': 'tool_call',
                                    'id': 'srv_use_cass_001',
                                    'name': 'advisor',
                                    'arguments': {'question': 'Is 2+2 = 4?'},
                                },
                                {
                                    'type': 'tool_call_response',
                                    'id': 'srv_use_cass_001',
                                    'response': {'type': 'advisor_tool_result', 'summary': 'Yes, 2+2=4.'},
                                },
                                {'type': 'text', 'content': '4'},
                            ],
                        }
                    ],
                    'logfire.msg': 'chat claude-sonnet-4-6',
                    'logfire.span_type': 'span',
                    'gen_ai.usage.partial.input_tokens': 9344,
                    'gen_ai.usage.partial.output_tokens': 1,
                    'gen_ai.usage.partial.cache_read.input_tokens': 7166,
                    'gen_ai.usage.partial.cache_creation.input_tokens': 2175,
                    'gen_ai.response.id': 'msg_01SrvToolFixturePpppppppppp',
                    'claude.message.uuid': 'f85bae42-7165-4a5e-991b-68c4fd586f8a',
                    'gen_ai.request.model': 'claude-sonnet-4-6',
                    'gen_ai.response.model': 'claude-sonnet-4-6',
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'gen_ai.operation.name': {},
                            'gen_ai.provider.name': {},
                            'gen_ai.system': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'gen_ai.output.messages': {'type': 'array'},
                            'gen_ai.system_instructions': {'type': 'array'},
                            'gen_ai.request.model': {},
                            'gen_ai.response.model': {},
                            'gen_ai.usage.partial.input_tokens': {},
                            'gen_ai.usage.partial.output_tokens': {},
                            'gen_ai.usage.partial.cache_read.input_tokens': {},
                            'gen_ai.usage.partial.cache_creation.input_tokens': {},
                            'gen_ai.response.id': {},
                            'claude.message.uuid': {},
                        },
                    },
                },
            },
            {
                'name': 'invoke_agent',
                'context': {'trace_id': 2, 'span_id': 3, 'is_remote': False},
                'parent': None,
                'start_time': 3000000000,
                'end_time': 6000000000,
                'attributes': {
                    'code.filepath': 'test_claude_agent_sdk.py',
                    'code.function': 'test_server_tool_blocks_cassette',
                    'code.lineno': 123,
                    'gen_ai.operation.name': 'invoke_agent',
                    'gen_ai.provider.name': 'anthropic',
                    'gen_ai.system': 'anthropic',
                    'gen_ai.agent.name': 'claude-code',
                    'gen_ai.input.messages': [{'role': 'user', 'parts': [{'type': 'text', 'content': 'What is 2+2?'}]}],
                    'gen_ai.system_instructions': [{'type': 'text', 'content': 'Be helpful'}],
                    'logfire.msg_template': 'invoke_agent',
                    'logfire.msg': 'invoke_agent',
                    'logfire.span_type': 'span',
                    'gen_ai.usage.input_tokens': 9344,
                    'gen_ai.usage.output_tokens': 5,
                    'gen_ai.usage.cache_read.input_tokens': 7166,
                    'gen_ai.usage.cache_creation.input_tokens': 2175,
                    'operation.cost': 0.01039005,
                    'gen_ai.conversation.id': '7ed3c21d-374b-491a-8c66-05e191f6a0be',
                    'num_turns': 1,
                    'duration_ms': 2263,
                    'duration_api_ms': 2257,
                    'claude.result.subtype': 'success',
                    'claude.result.text': '4',
                    'gen_ai.response.finish_reasons': ['end_turn'],
                    'claude.model_usage': {
                        'claude-sonnet-4-6': {
                            'inputTokens': 3,
                            'outputTokens': 5,
                            'cacheReadInputTokens': 7166,
                            'cacheCreationInputTokens': 2175,
                            'webSearchRequests': 0,
                            'costUSD': 0.01039005,
                            'contextWindow': 200000,
                            'maxOutputTokens': 32000,
                        }
                    },
                    'gen_ai.request.model': 'claude-sonnet-4-6',
                    'gen_ai.response.model': 'claude-sonnet-4-6',
                    'pydantic_ai.all_messages': [
                        {'role': 'user', 'parts': [{'type': 'text', 'content': 'What is 2+2?'}]},
                        {
                            'role': 'assistant',
                            'parts': [
                                {'type': 'text', 'content': 'Checking with advisor.'},
                                {
                                    'type': 'tool_call',
                                    'id': 'srv_use_cass_001',
                                    'name': 'advisor',
                                    'arguments': {'question': 'Is 2+2 = 4?'},
                                },
                                {
                                    'type': 'tool_call_response',
                                    'id': 'srv_use_cass_001',
                                    'response': {'type': 'advisor_tool_result', 'summary': 'Yes, 2+2=4.'},
                                },
                                {'type': 'text', 'content': '4'},
                            ],
                        },
                    ],
                    'claude.tools_used': [{'tool': 'advisor', 'count': 1}],
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'gen_ai.operation.name': {},
                            'gen_ai.provider.name': {},
                            'gen_ai.system': {},
                            'gen_ai.agent.name': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'gen_ai.system_instructions': {'type': 'array'},
                            'gen_ai.usage.input_tokens': {},
                            'gen_ai.usage.output_tokens': {},
                            'gen_ai.usage.cache_read.input_tokens': {},
                            'gen_ai.usage.cache_creation.input_tokens': {},
                            'operation.cost': {},
                            'gen_ai.conversation.id': {},
                            'num_turns': {},
                            'duration_ms': {},
                            'duration_api_ms': {},
                            'claude.result.subtype': {},
                            'claude.result.text': {},
                            'gen_ai.response.finish_reasons': {'type': 'array'},
                            'claude.model_usage': {'type': 'object'},
                            'gen_ai.request.model': {},
                            'gen_ai.response.model': {},
                            'pydantic_ai.all_messages': {'type': 'array'},
                            'claude.tools_used': {'type': 'array'},
                        },
                    },
                },
            },
            {
                'name': 'disconnect',
                'context': {'trace_id': 3, 'span_id': 7, 'is_remote': False},
                'parent': None,
                'start_time': 7000000000,
                'end_time': 8000000000,
                'attributes': {
                    'code.filepath': 'test_claude_agent_sdk.py',
                    'code.function': 'test_server_tool_blocks_cassette',
                    'code.lineno': 123,
                    'logfire.msg_template': 'disconnect',
                    'logfire.span_type': 'span',
                    'logfire.msg': 'disconnect',
                },
            },
        ]
    )


@pytest.mark.anyio
async def test_ratelimit_and_mirror_error_cassette(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, exporter: TestExporter
) -> None:
    """RateLimitEvent and MirrorErrorMessage in the stream produce log spans.

    The cassette is basic_conversation.json with two extra ``recv`` entries
    spliced in right after ``system/init``: a ``rate_limit_event`` with
    status ``allowed_warning`` and a ``system/mirror_error``. This exercises
    the branches added for logfire issue #2.
    """
    record = request.config.getoption('--record-claude-cassettes', default=False)
    # Recording is not meaningful for this cassette — mirror_error is
    # SDK-synthesized (not emitted by the CLI) and rate_limit_event requires
    # a real rate-limit transition. The cassette is hand-assembled; skip the
    # record path so CI never tries to overwrite it.
    if record:  # pragma: no cover
        pytest.skip('ratelimit_mirror_conversation.json is hand-assembled, not recorded')

    import claude_agent_sdk as _sdk

    if not hasattr(_sdk, 'RateLimitEvent') or not hasattr(_sdk, 'MirrorErrorMessage'):
        pytest.skip('SDK version lacks RateLimitEvent / MirrorErrorMessage — needs claude-agent-sdk >= 0.1.62')

    client = _make_client('ratelimit_mirror_conversation.json', monkeypatch=monkeypatch)
    try:
        await client.connect()
        await client.query('What is 2+2?')
        collected = [msg async for msg in client.receive_response()]
    finally:
        await _close_sdk_streams(client)
        await client.disconnect()

    # SDK itself should have yielded both messages through to the caller.
    assert any(isinstance(m, _sdk.RateLimitEvent) for m in collected), 'SDK did not yield RateLimitEvent'
    assert any(isinstance(m, _sdk.MirrorErrorMessage) for m in collected), 'SDK did not yield MirrorErrorMessage'

    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'connect',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'logfire.span_type': 'span',
                    'logfire.msg_template': 'connect',
                    'claude.cli_path': IsStr(regex=r'.*/fake_claude\.py$'),
                    'logfire.msg': 'connect',
                    'code.filepath': 'test_claude_agent_sdk.py',
                    'code.function': 'test_ratelimit_and_mirror_error_cassette',
                    'code.lineno': 123,
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {'claude.cli_path': {}},
                    },
                },
            },
            {
                'name': 'rate_limit {status}',
                'context': {'trace_id': 2, 'span_id': 7, 'is_remote': False},
                'parent': {'trace_id': 2, 'span_id': 3, 'is_remote': False},
                'start_time': 5000000000,
                'end_time': 5000000000,
                'attributes': {
                    'logfire.span_type': 'log',
                    'logfire.level_num': 13,
                    'logfire.msg_template': 'rate_limit {status}',
                    'logfire.msg': 'rate_limit allowed_warning',
                    'code.filepath': 'test_claude_agent_sdk.py',
                    'code.function': 'test_ratelimit_and_mirror_error_cassette',
                    'code.lineno': 123,
                    'status': 'allowed_warning',
                    'gen_ai.rate_limit.status': 'allowed_warning',
                    'gen_ai.rate_limit.resets_at': 1800000000,
                    'gen_ai.rate_limit.type': 'five_hour',
                    'gen_ai.rate_limit.utilization': 0.87,
                    'gen_ai.rate_limit.raw': {
                        'status': 'allowed_warning',
                        'resetsAt': 1800000000,
                        'rateLimitType': 'five_hour',
                        'utilization': 0.87,
                    },
                    'gen_ai.conversation.id': '7ed3c21d-374b-491a-8c66-05e191f6a0be',
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'status': {},
                            'gen_ai.rate_limit.status': {},
                            'gen_ai.rate_limit.resets_at': {},
                            'gen_ai.rate_limit.type': {},
                            'gen_ai.rate_limit.utilization': {},
                            'gen_ai.rate_limit.raw': {'type': 'object'},
                            'gen_ai.conversation.id': {},
                        },
                    },
                },
            },
            {
                'name': 'mirror store error: {error}',
                'context': {'trace_id': 2, 'span_id': 8, 'is_remote': False},
                'parent': {'trace_id': 2, 'span_id': 3, 'is_remote': False},
                'start_time': 6000000000,
                'end_time': 6000000000,
                'attributes': {
                    'code.filepath': 'test_claude_agent_sdk.py',
                    'logfire.level_num': 17,
                    'code.function': 'test_ratelimit_and_mirror_error_cassette',
                    'code.lineno': 123,
                    'error': 'store append timed out',
                    'error.type': 'MirrorError',
                    'gen_ai.conversation.id': '7ed3c21d-374b-491a-8c66-05e191f6a0be',
                    'logfire.msg_template': 'mirror store error: {error}',
                    'logfire.span_type': 'log',
                    'logfire.msg': 'mirror store error: store append timed out',
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'error': {},
                            'error.type': {},
                            'gen_ai.conversation.id': {},
                        },
                    },
                },
            },
            {
                'name': 'chat claude-sonnet-4-6',
                'context': {'trace_id': 2, 'span_id': 5, 'is_remote': False},
                'parent': {'trace_id': 2, 'span_id': 3, 'is_remote': False},
                'start_time': 4000000000,
                'end_time': 7000000000,
                'attributes': {
                    'code.filepath': 'test_claude_agent_sdk.py',
                    'code.function': 'test_ratelimit_and_mirror_error_cassette',
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.provider.name': 'anthropic',
                    'gen_ai.system': 'anthropic',
                    'gen_ai.input.messages': [{'role': 'user', 'parts': [{'type': 'text', 'content': 'What is 2+2?'}]}],
                    'gen_ai.system_instructions': [{'type': 'text', 'content': 'Be helpful'}],
                    'code.lineno': 123,
                    'gen_ai.output.messages': [{'role': 'assistant', 'parts': [{'type': 'text', 'content': '4'}]}],
                    'gen_ai.request.model': 'claude-sonnet-4-6',
                    'gen_ai.response.model': 'claude-sonnet-4-6',
                    'gen_ai.usage.partial.input_tokens': 9344,
                    'gen_ai.usage.partial.output_tokens': 1,
                    'gen_ai.usage.partial.cache_read.input_tokens': 7166,
                    'gen_ai.usage.partial.cache_creation.input_tokens': 2175,
                    'gen_ai.response.id': 'msg_01BxK8UH2LyuFLVXSPamqHam',
                    'claude.message.uuid': 'ce1101bc-27c6-41b1-ae16-5edf06b0927c',
                    'logfire.msg_template': 'chat',
                    'logfire.msg': 'chat claude-sonnet-4-6',
                    'logfire.span_type': 'span',
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'gen_ai.operation.name': {},
                            'gen_ai.provider.name': {},
                            'gen_ai.system': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'gen_ai.system_instructions': {'type': 'array'},
                            'gen_ai.output.messages': {'type': 'array'},
                            'gen_ai.request.model': {},
                            'gen_ai.response.model': {},
                            'gen_ai.usage.partial.input_tokens': {},
                            'gen_ai.usage.partial.output_tokens': {},
                            'gen_ai.usage.partial.cache_read.input_tokens': {},
                            'gen_ai.usage.partial.cache_creation.input_tokens': {},
                            'gen_ai.response.id': {},
                            'claude.message.uuid': {},
                        },
                    },
                },
            },
            {
                'name': 'invoke_agent',
                'context': {'trace_id': 2, 'span_id': 3, 'is_remote': False},
                'parent': None,
                'start_time': 3000000000,
                'end_time': 8000000000,
                'attributes': {
                    'logfire.span_type': 'span',
                    'logfire.msg_template': 'invoke_agent',
                    'gen_ai.operation.name': 'invoke_agent',
                    'gen_ai.provider.name': 'anthropic',
                    'gen_ai.system': 'anthropic',
                    'gen_ai.agent.name': 'claude-code',
                    'gen_ai.input.messages': [{'role': 'user', 'parts': [{'type': 'text', 'content': 'What is 2+2?'}]}],
                    'gen_ai.system_instructions': [{'type': 'text', 'content': 'Be helpful'}],
                    'logfire.msg': 'invoke_agent',
                    'code.filepath': 'test_claude_agent_sdk.py',
                    'code.function': 'test_ratelimit_and_mirror_error_cassette',
                    'gen_ai.usage.input_tokens': 9344,
                    'gen_ai.usage.output_tokens': 5,
                    'gen_ai.usage.cache_read.input_tokens': 7166,
                    'gen_ai.usage.cache_creation.input_tokens': 2175,
                    'operation.cost': 0.01039005,
                    'code.lineno': 123,
                    'num_turns': 1,
                    'duration_ms': 2263,
                    'duration_api_ms': 2257,
                    'claude.result.subtype': 'success',
                    'claude.result.text': '4',
                    'gen_ai.response.finish_reasons': ['end_turn'],
                    'claude.model_usage': {
                        'claude-sonnet-4-6': {
                            'inputTokens': 3,
                            'outputTokens': 5,
                            'cacheReadInputTokens': 7166,
                            'cacheCreationInputTokens': 2175,
                            'webSearchRequests': 0,
                            'costUSD': 0.01039005,
                            'contextWindow': 200000,
                            'maxOutputTokens': 32000,
                        }
                    },
                    'gen_ai.request.model': 'claude-sonnet-4-6',
                    'gen_ai.response.model': 'claude-sonnet-4-6',
                    'pydantic_ai.all_messages': [
                        {'role': 'user', 'parts': [{'type': 'text', 'content': 'What is 2+2?'}]},
                        {'role': 'assistant', 'parts': [{'type': 'text', 'content': '4'}]},
                    ],
                    'gen_ai.conversation.id': '7ed3c21d-374b-491a-8c66-05e191f6a0be',
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'gen_ai.operation.name': {},
                            'gen_ai.provider.name': {},
                            'gen_ai.system': {},
                            'gen_ai.agent.name': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'gen_ai.system_instructions': {'type': 'array'},
                            'gen_ai.usage.input_tokens': {},
                            'gen_ai.usage.output_tokens': {},
                            'gen_ai.usage.cache_read.input_tokens': {},
                            'gen_ai.usage.cache_creation.input_tokens': {},
                            'operation.cost': {},
                            'gen_ai.conversation.id': {},
                            'num_turns': {},
                            'duration_ms': {},
                            'duration_api_ms': {},
                            'claude.result.subtype': {},
                            'claude.result.text': {},
                            'gen_ai.response.finish_reasons': {'type': 'array'},
                            'claude.model_usage': {'type': 'object'},
                            'gen_ai.request.model': {},
                            'gen_ai.response.model': {},
                            'pydantic_ai.all_messages': {'type': 'array'},
                        },
                    },
                },
            },
            {
                'name': 'disconnect',
                'context': {'trace_id': 3, 'span_id': 9, 'is_remote': False},
                'parent': None,
                'start_time': 9000000000,
                'end_time': 10000000000,
                'attributes': {
                    'code.filepath': 'test_claude_agent_sdk.py',
                    'code.function': 'test_ratelimit_and_mirror_error_cassette',
                    'code.lineno': 123,
                    'logfire.msg_template': 'disconnect',
                    'logfire.span_type': 'span',
                    'logfire.msg': 'disconnect',
                },
            },
        ]
    )


@pytest.mark.anyio
async def test_rate_limit_rejected_sets_error_level(exporter: TestExporter) -> None:
    """A ``rate_limit_event`` with status 'rejected' escalates the root span."""
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent') as root:
        msg = SimpleNamespace(
            rate_limit_info=SimpleNamespace(
                status='rejected',
                resets_at=1800000000,
                rate_limit_type='five_hour',
                utilization=1.0,
                overage_status=None,
                overage_resets_at=None,
                overage_disabled_reason=None,
            ),
            session_id='sess-abc',
        )
        _record_rate_limit_event(logfire_instance, root, msg)  # pyright: ignore[reportArgumentType]

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    log_spans = [s for s in spans if s['name'] == 'rate_limit {status}']
    assert len(log_spans) == 1
    log = log_spans[0]
    assert log['attributes']['logfire.level_num'] == 17  # error
    assert log['attributes']['error.type'] == 'RateLimitRejected'

    # Root span itself is escalated to error via set_level.
    root_spans = [s for s in spans if s['name'] == 'invoke_agent']
    assert len(root_spans) == 1
    assert root_spans[0]['attributes']['logfire.level_num'] == 17


@pytest.mark.anyio
async def test_mirror_error_without_key(exporter: TestExporter) -> None:
    """MirrorErrorMessage with key=None (not every mirror error has one)."""
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE.with_settings(custom_scope_suffix='claude_agent_sdk')
    with logfire_instance.span('invoke_agent'):
        _record_mirror_error(logfire_instance, SimpleNamespace(error='oh no', key=None))  # pyright: ignore[reportArgumentType]

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    mirror = next(s for s in spans if s['name'] == 'mirror store error: {error}')
    assert mirror['attributes']['error.type'] == 'MirrorError'
    assert 'gen_ai.conversation.id' not in mirror['attributes']
