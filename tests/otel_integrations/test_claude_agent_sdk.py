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
from unittest.mock import Mock

import pytest

pytest.importorskip('claude_agent_sdk', reason='claude_agent_sdk requires Python 3.10+')

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    ServerToolResultBlock,
    ServerToolUseBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from claude_agent_sdk.types import HookContext
from dirty_equals import IsStr
from inline_snapshot import snapshot

import logfire
from logfire._internal.integrations.claude_agent_sdk import (
    _clear_state,
    _content_blocks_to_output_messages,
    _ConversationState,
    _extract_usage,
    _inject_tracing_hooks,
    _record_mirror_error,
    _record_rate_limit_event,
    _record_result,
    _set_state,
    post_tool_use_failure_hook,
    post_tool_use_hook,
    pre_tool_use_hook,
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
        history = state._history  # pyright: ignore[reportPrivateUsage]
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
    options = ClaudeAgentOptions(hooks=None)
    _inject_tracing_hooks(options)
    assert options.hooks is not None
    assert 'PreToolUse' in options.hooks
    assert len(options.hooks['PreToolUse']) == 1


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
    assert len(options.hooks['PreToolUse']) == 2
    assert options.hooks['PreToolUse'][1] is existing_hook


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
                'name': 'chat claude-sonnet-4-6',
                'context': {'trace_id': 1, 'span_id': 3, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'start_time': 2000000000,
                'end_time': 3000000000,
                'attributes': {
                    'code.filepath': 'test_claude_agent_sdk.py',
                    'code.function': 'test_basic_conversation_cassette',
                    'code.lineno': 123,
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.provider.name': 'anthropic',
                    'gen_ai.system': 'anthropic',
                    'gen_ai.response.model': 'claude-sonnet-4-6',
                    'gen_ai.system_instructions': [{'type': 'text', 'content': 'Be helpful'}],
                    'gen_ai.output.messages': [{'role': 'assistant', 'parts': [{'type': 'text', 'content': '4'}]}],
                    'gen_ai.input.messages': [{'role': 'user', 'parts': [{'type': 'text', 'content': 'What is 2+2?'}]}],
                    'gen_ai.usage.partial.input_tokens': 9344,
                    'gen_ai.request.model': 'claude-sonnet-4-6',
                    'gen_ai.usage.partial.output_tokens': 1,
                    'gen_ai.usage.partial.cache_read.input_tokens': 7166,
                    'gen_ai.usage.partial.cache_creation.input_tokens': 2175,
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
                'end_time': 4000000000,
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
                    'gen_ai.response.finish_reasons': ['end_turn'],
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
                            'claude.model_usage': {'type': 'object'},
                            'gen_ai.response.finish_reasons': {'type': 'array'},
                            'gen_ai.request.model': {},
                            'gen_ai.response.model': {},
                            'pydantic_ai.all_messages': {'type': 'array'},
                        },
                    },
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
                'name': 'chat claude-sonnet-4-6',
                'context': {'trace_id': 1, 'span_id': 3, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'start_time': 2000000000,
                'end_time': 3000000000,
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
                    'logfire.span_type': 'span',
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
                    'gen_ai.request.model': 'claude-sonnet-4-6',
                    'gen_ai.response.model': 'claude-sonnet-4-6',
                    'logfire.msg': 'chat claude-sonnet-4-6',
                    'gen_ai.usage.partial.input_tokens': 9344,
                    'gen_ai.usage.partial.output_tokens': 1,
                    'gen_ai.usage.partial.cache_read.input_tokens': 7166,
                    'gen_ai.usage.partial.cache_creation.input_tokens': 2175,
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
                },
            },
            {
                'name': 'invoke_agent',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 4000000000,
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
                    'gen_ai.response.finish_reasons': ['end_turn'],
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
                            'claude.model_usage': {'type': 'object'},
                            'gen_ai.response.finish_reasons': {'type': 'array'},
                            'gen_ai.request.model': {},
                            'gen_ai.response.model': {},
                            'pydantic_ai.all_messages': {'type': 'array'},
                            'claude.tools_used': {'type': 'array'},
                        },
                    },
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
                'name': 'rate_limit {status}',
                'context': {'trace_id': 1, 'span_id': 5, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'start_time': 3000000000,
                'end_time': 3000000000,
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
                'context': {'trace_id': 1, 'span_id': 6, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'start_time': 4000000000,
                'end_time': 4000000000,
                'attributes': {
                    'logfire.span_type': 'log',
                    'logfire.level_num': 17,
                    'logfire.msg_template': 'mirror store error: {error}',
                    'logfire.msg': 'mirror store error: store append timed out',
                    'code.filepath': 'test_claude_agent_sdk.py',
                    'code.function': 'test_ratelimit_and_mirror_error_cassette',
                    'code.lineno': 123,
                    'error': 'store append timed out',
                    'error.type': 'MirrorError',
                    'gen_ai.conversation.id': '7ed3c21d-374b-491a-8c66-05e191f6a0be',
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {'error': {}, 'error.type': {}, 'gen_ai.conversation.id': {}},
                    },
                },
            },
            {
                'name': 'chat claude-sonnet-4-6',
                'context': {'trace_id': 1, 'span_id': 3, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'start_time': 2000000000,
                'end_time': 5000000000,
                'attributes': {
                    'code.filepath': 'test_claude_agent_sdk.py',
                    'code.function': 'test_ratelimit_and_mirror_error_cassette',
                    'code.lineno': 123,
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.provider.name': 'anthropic',
                    'gen_ai.system': 'anthropic',
                    'gen_ai.input.messages': [{'role': 'user', 'parts': [{'type': 'text', 'content': 'What is 2+2?'}]}],
                    'gen_ai.system_instructions': [{'type': 'text', 'content': 'Be helpful'}],
                    'logfire.msg_template': 'chat',
                    'logfire.span_type': 'span',
                    'gen_ai.output.messages': [{'role': 'assistant', 'parts': [{'type': 'text', 'content': '4'}]}],
                    'gen_ai.request.model': 'claude-sonnet-4-6',
                    'gen_ai.response.model': 'claude-sonnet-4-6',
                    'logfire.msg': 'chat claude-sonnet-4-6',
                    'gen_ai.usage.partial.input_tokens': 9344,
                    'gen_ai.usage.partial.output_tokens': 1,
                    'gen_ai.usage.partial.cache_read.input_tokens': 7166,
                    'gen_ai.usage.partial.cache_creation.input_tokens': 2175,
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
                },
            },
            {
                'name': 'invoke_agent',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 6000000000,
                'attributes': {
                    'code.filepath': 'test_claude_agent_sdk.py',
                    'code.function': 'test_ratelimit_and_mirror_error_cassette',
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
                    'gen_ai.response.finish_reasons': ['end_turn'],
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
                            'claude.model_usage': {'type': 'object'},
                            'gen_ai.response.finish_reasons': {'type': 'array'},
                            'gen_ai.request.model': {},
                            'gen_ai.response.model': {},
                            'pydantic_ai.all_messages': {'type': 'array'},
                        },
                    },
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
        _record_rate_limit_event(logfire_instance, root, msg)

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
        _record_mirror_error(logfire_instance, SimpleNamespace(error='oh no', key=None))

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    mirror = next(s for s in spans if s['name'] == 'mirror store error: {error}')
    assert mirror['attributes']['error.type'] == 'MirrorError'
    assert 'gen_ai.conversation.id' not in mirror['attributes']
