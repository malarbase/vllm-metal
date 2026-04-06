# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-metal project
"""vLLM tool parser for Google Gemma 4 chat templates.

Gemma 4 emits tool calls as::

    <|tool_call>call:tool_name{arg:<|"|>value<|"|>}<tool_call|>

Upstream vLLM 0.17 registers ``functiongemma`` for FunctionGemma models, which
use ``<start_function_call>`` / ``<escape>`` — a different wire format.  mlx-lm
parses the Gemma 4 format in ``mlx_lm/tool_parsers/gemma4.py``; this module
ports that logic for vLLM's OpenAI API layer.

See: https://github.com/ml-explore/mlx-lm/commit/3257c3df172977c97fdfe3740e3a5edeb812e0b5
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence

import regex as re
from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import ToolParser

logger = logging.getLogger(__name__)

_TOOL_CALL_START = "<|tool_call>"
_TOOL_CALL_END = "<tool_call|>"
_TOOL_CALL_BODY = re.compile(r"call:(\w+)(\{.*\})", re.DOTALL)
_STRING_DELIM = re.compile(r'<\|"\|>(.*?)<\|"\|>', re.DOTALL)


def _gemma4_args_to_json(text: str) -> str:
    """Convert Gemma 4 braced args to JSON (ported from mlx-lm gemma4 parser)."""
    strings: list[str] = []

    def _capture(m: re.Match[str]) -> str:
        strings.append(m.group(1))
        return f"\x00{len(strings) - 1}\x00"

    out = _STRING_DELIM.sub(_capture, text)
    out = re.sub(r"(?<=[{,])(\w+):", r'"\1":', out)
    for i, s in enumerate(strings):
        out = out.replace(f"\x00{i}\x00", json.dumps(s))
    return out


def _parse_inner(inner: str) -> ExtractedToolCallInformation:
    inner = inner.strip()
    m = _TOOL_CALL_BODY.search(inner)
    if not m:
        return ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=inner
        )
    func_name = m.group(1)
    args_str = m.group(2)
    try:
        json_str = _gemma4_args_to_json(args_str)
        arguments = json.loads(json_str)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Gemma4 tool arg parse failed: %s", e)
        return ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=inner
        )
    tc = ToolCall(
        type="function",
        function=FunctionCall(
            name=func_name,
            arguments=json.dumps(arguments, ensure_ascii=False),
        ),
    )
    return ExtractedToolCallInformation(
        tools_called=True, tool_calls=[tc], content=None
    )


class Gemma4ToolParser(ToolParser):
    """Parse Gemma 4 ``<|tool_call>`` … ``<tool_call|>`` tool invocations."""

    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)
        self._gemma4_stream_emitted = False

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            request.skip_special_tokens = False
        return request

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        text = model_output.strip()
        if _TOOL_CALL_START in text:
            start = text.find(_TOOL_CALL_START)
            end = text.find(_TOOL_CALL_END, start)
            if end < 0:
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )
            inner = text[start + len(_TOOL_CALL_START) : end].strip()
            ext = _parse_inner(inner)
            if ext.tools_called:
                lead = text[:start].strip()
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=ext.tool_calls,
                    content=lead or None,
                )
            return ext
        return _parse_inner(text)

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        if self._gemma4_stream_emitted:
            return DeltaMessage(content=delta_text) if delta_text else None

        if _TOOL_CALL_END not in current_text:
            if _TOOL_CALL_START in current_text:
                return None
            return DeltaMessage(content=delta_text) if delta_text else None

        ext = self.extract_tool_calls(current_text, request)
        self._gemma4_stream_emitted = True

        if ext.tools_called and ext.tool_calls:
            dtc: list[DeltaToolCall] = []
            for i, tc in enumerate(ext.tool_calls):
                dtc.append(
                    DeltaToolCall(
                        index=i,
                        type="function",
                        id=make_tool_call_id(),
                        function=DeltaFunctionCall(
                            name=tc.function.name,
                            arguments=tc.function.arguments or "{}",
                        ).model_dump(exclude_none=True),
                    )
                )
            return DeltaMessage(tool_calls=dtc, content="")

        return DeltaMessage(content=current_text) if current_text else None
