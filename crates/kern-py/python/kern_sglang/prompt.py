"""A request's prompt as the token ids SGLang's own server would schedule.

kern's prefill answers SGLang's router for `/generate`, `/v1/completions`
and `/v1/chat/completions`, and the decode side tokenizes the same request
itself, so the prefill must reach the very ids the decode side does: the
chat template and its kwargs, the reasoning and tool-call parsers, a model's
own chat encoding. Nothing here reimplements any of it. The half of SGLang's
`TokenizerManager` that loads the model config, the tokenizer and the
processor runs as SGLang runs it (none of its process or IPC); its
`TemplateManager` loads the chat template and the auto parsers resolve as
the engine resolves them; SGLang's OpenAI handlers turn a request into the
`GenerateReqInput` they would schedule, and text left in it is encoded as
the tokenizer manager encodes it. What shapes all of this is the server
arguments the stage published (`kern_sglang.prefill.Stage`), which must be
the decode side's.

Called from the head's HTTP threads, holding the GIL for the call.
"""
import json

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest, CompletionRequest
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.parser.template_manager import TemplateManager
from sglang.srt.runtime_context import get_model, get_serving, get_server_args


class Prompts:
    def __init__(self):
        tm = TokenizerManager.__new__(TokenizerManager)
        tm.server_args = get_server_args()
        tm.init_model_config()
        tm.init_tokenizer_and_processor()
        templates = TemplateManager()
        templates.initialize_templates(tokenizer_manager=tm, model_path=get_model().model_path,
                                       chat_template=get_serving().chat_template,
                                       completion_template=get_serving().completion_template)
        for attr, suggested in (("reasoning_parser", templates.suggested_reasoning_parser),
                                ("tool_call_parser", templates.suggested_tool_call_parser)):
            if tm.config_value(attr) == "auto":
                tm.record_config_updates("template-detection", **{attr: suggested})
        self.tokenizer = tm.tokenizer
        self.chat = OpenAIServingChat(tm, templates)
        self.completion = OpenAIServingCompletion(tm, templates)

    def ids(self, route: str, body: str) -> list[int]:
        """The prompt of a request to `route` (`chat`, `completion` or
        `generate`), given as its JSON body."""
        b = json.loads(body)
        if route == "chat":
            r, _ = self.chat._convert_to_internal_request(ChatCompletionRequest(**b))
        elif route == "completion":
            r, _ = self.completion._convert_to_internal_request(CompletionRequest(**b))
        elif route == "generate":
            r = b
        else:
            raise ValueError(f"no prompt for route `{route}`")
        ids, text = (r.get("input_ids"), r.get("text")) if isinstance(r, dict) else (r.input_ids, r.text)
        if ids is not None:
            if not ids or isinstance(ids[0], list):
                raise ValueError("a prefill takes one prompt")
            return [int(t) for t in ids]
        if not isinstance(text, str):
            raise ValueError("a prefill takes one prompt, as `text` or `input_ids`")
        if getattr(self.tokenizer, "is_fast", False):
            return list(self.tokenizer([text])["input_ids"][0])
        return list(self.tokenizer.encode(text))
