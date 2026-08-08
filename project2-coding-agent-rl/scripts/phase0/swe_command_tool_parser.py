"""Custom vLLM tool parser for SWE-Master-style `<command>` output.

SWE-Master-4B-RL is RL-trained to emit SWE-agent-style actions:

    <execute>
    <command>ls -la</command>
    </execute>

(or a bare ``<command>...</command>``, or a ```bash fenced block).  The stock
``qwen3_xml`` parser expects Qwen3's ``<tool_call>`` XML and therefore misses
these.  This parser converts the first matched action into a single ``bash``
tool call so the mini-swe-agent toolcall protocol works unchanged.
"""
import json
import re

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
)

logger = init_logger(__name__)

TOOL_CALL_PATTERNS = [
    # <execute>...<command>...</command>...</execute>
    re.compile(r"<execute>\s*<command>(.*?)</command>\s*</execute>", re.S),
    # bare <command>...</command>
    re.compile(r"<command>(.*?)</command>", re.S),
    # ```bash ... ``` fenced block
    re.compile(r"```(?:bash|sh)\s*\n(.*?)```", re.S),
    # model sometimes imitates the OpenAI JSON tool-call format it sees in history
    re.compile(
        r"\{[\"']name[\"']:\s*[\"']bash[\"']\s*,\s*[\"']arguments[\"']:\s*\{[\"']command[\"']:\s*[\"'](.*?)[\"']\s*\}\}",
        re.S,
    ),
]

# The RL-trained model often continues generating the tool observation itself
# after </command> (SWE-agent transcript style: "Exit code: 0 ..."), sometimes
# followed by a whole second command cycle. Everything from this marker onward
# is the model's own hallucinated transcript, not content we want in history.
TOOL_RESPONSE_MARKER = "<tool_response>"


class SweCommandToolParser(ToolParser):
    """Extract SWE-agent-style ``<command>`` actions as a ``bash`` tool call."""

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)
        self.tool_names = [
            t.function.name if hasattr(t, "function") else getattr(t, "name", None)
            for t in (tools or [])
        ]
        self.prev_tool_call_arr: list[dict] = []
        self.streamed_args_for_tool: list[str] = []

        logger.info("vLLM Successfully import tool parser %s !", self.__class__.__name__)

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        self.prev_tool_call_arr = []
        self.streamed_args_for_tool = []
        content = model_output
        cmd = None
        for pat in TOOL_CALL_PATTERNS:
            m = pat.search(model_output)
            if m:
                cmd = m.group(1).strip()
                break
        if cmd is None:
            return ExtractedToolCallInformation(
                tool_calls=[],
                tools_called=False,
                content=content,
            )
        # Drop the model's self-generated transcript tail (fake observations the
        # RL model writes after </command>, with or without <tool_response>
        # wrappers) so the stored assistant content stays clean and does not
        # pollute next context. The command itself comes first, so cut at the
        # first observation-looking line after the match.
        cut = content.find(TOOL_RESPONSE_MARKER)
        if cut == -1:
            cut = re.search(r"\nExit code: \d+", content[m.end():])
            if cut:
                cut = m.end() + cut.start()
        if cut is not None and cut != -1:
            content = content[:cut].rstrip()
        tool_calls = [
            ToolCall(
                id=make_tool_call_id(),
                type="function",
                function=FunctionCall(
                    name="bash",
                    arguments=json.dumps({"command": cmd}),
                ),
            )
        ]
        return ExtractedToolCallInformation(
            tool_calls=tool_calls,
            tools_called=True,
            content=content,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: list[int],
        current_token_ids: list[int],
        delta_token_ids: list[int],
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        # Non-streaming only; fall back to full extraction on the current text.
        return self.extract_tool_calls(current_text, request)
