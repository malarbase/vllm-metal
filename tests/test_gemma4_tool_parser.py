"""Tests for Gemma 4 OpenAI tool-call parsing (vllm-metal)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("vllm")

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

from vllm_metal.compat import apply_compat_patches
from vllm_metal.tool_parsers.gemma4_tool_parser import Gemma4ToolParser


@pytest.fixture
def tokenizer() -> MagicMock:
    tok = MagicMock()
    tok.get_vocab.return_value = {}
    return tok


@pytest.fixture
def parser(tokenizer: MagicMock) -> Gemma4ToolParser:
    return Gemma4ToolParser(tokenizer)


@pytest.fixture
def request_minimal() -> ChatCompletionRequest:
    return ChatCompletionRequest.model_construct(
        model="google/gemma-4-E4B-it",
        messages=[],
    )


def test_extract_tool_calls_wrapped_gemma4(parser: Gemma4ToolParser, request_minimal: ChatCompletionRequest) -> None:
    raw = '<|tool_call>call:read_file{path:<|"|>note.txt<|"|>}<tool_call|>'
    ext = parser.extract_tool_calls(raw, request_minimal)
    assert ext.tools_called is True
    assert len(ext.tool_calls) == 1
    assert ext.tool_calls[0].function.name == "read_file"
    assert "note.txt" in ext.tool_calls[0].function.arguments


def test_gemma4_parser_registered_after_compat() -> None:
    apply_compat_patches()
    from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

    cls = ToolParserManager.get_tool_parser("gemma4")
    assert cls is Gemma4ToolParser
