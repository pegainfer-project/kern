//! A chat conversation as the prompt kern-serve would feed: OpenAI JSON
//! into vLLM's `ChatRequest` the way vLLM's server converts it, rendered
//! by the renderer the checkpoint's `config.json` selects, tokenized
//! without added special tokens. kern-serve reaches the same renderer
//! through pegainfer; the conversion is private to `vllm-server`, so its
//! rules are restated here: `reasoning_content` is an alias of
//! `reasoning`, empty reasoning is dropped, a tool call without
//! `arguments` has `{}`.

use std::path::Path;

use anyhow::{bail, Context, Result};
use serde_json::Value;
use vllm_chat::{
    AssistantContentBlock, AssistantToolCall, ChatContent, ChatContentPart, ChatMessage, ChatOptions, ChatRequest,
    ChatTool, DynChatRenderer, LoadModelBackendsOptions, ResolvedToolContext, SamplingParams,
};
use vllm_text::tokenizer::DynTokenizer;
use vllm_text::{Prompt, TextDecodeOptions};

/// A checkpoint's renderer and tokenizer.
pub struct Renderer {
    chat: DynChatRenderer,
    tokenizer: DynTokenizer,
}

impl Renderer {
    pub fn load(dir: &Path) -> Result<Renderer> {
        let id = dir.canonicalize().with_context(|| format!("checkpoint {}", dir.display()))?;
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
        let options = LoadModelBackendsOptions { language_model_only: true, ..Default::default() };
        let b = rt
            .block_on(vllm_chat::load_model_backends(&id.to_string_lossy(), options))
            .with_context(|| format!("loading the chat renderer of {}", dir.display()))?;
        Ok(Renderer { chat: b.chat_backend.chat_renderer(), tokenizer: b.text_backend.tokenizer() })
    }

    /// The prompt token ids of one conversation: OpenAI `messages` and,
    /// when it calls tools, their OpenAI `tools` schemas.
    pub fn prompt(&self, messages: &str, tools: Option<&str>) -> Result<Vec<i64>> {
        let messages: Vec<Value> = serde_json::from_str(messages).context("`messages` is not a JSON array")?;
        let messages = messages
            .iter()
            .enumerate()
            .map(|(i, m)| message(m).with_context(|| format!("message {i}")))
            .collect::<Result<Vec<_>>>()?;
        let tools: Vec<ChatTool> = match tools {
            Some(t) => {
                let list: Vec<Value> = serde_json::from_str(t).context("`tools` is not a JSON array")?;
                list.iter().map(tool).collect::<Result<_>>()?
            }
            None => Vec::new(),
        };
        let tool_context = ResolvedToolContext::new(&messages, tools, None, true)?;
        let request = ChatRequest {
            request_id: String::new(),
            messages,
            sampling_params: SamplingParams::default(),
            chat_options: ChatOptions::default(),
            tool_context,
            decode_options: TextDecodeOptions::default(),
            intermediate: false,
            prompt_truncation: None,
            priority: 0,
            documents: None,
            cache_salt: None,
            add_special_tokens: false,
            data_parallel_rank: None,
            session_id: None,
            lora_request: None,
        };
        request.validate()?;
        let ids = match self.chat.render(&request)?.prompt {
            Prompt::TokenIds(ids) => ids,
            Prompt::Text(text) => self.tokenizer.encode(&text, false)?,
        };
        Ok(ids.into_iter().map(i64::from).collect())
    }
}

fn message(m: &Value) -> Result<ChatMessage> {
    let role = m.get("role").and_then(Value::as_str).context("no `role`")?;
    let content = || content(m.get("content").unwrap_or(&Value::Null));
    Ok(match role {
        "system" => ChatMessage::system(content()?),
        "user" => ChatMessage::user(content()?),
        "developer" => ChatMessage::developer(content()?, None),
        "tool" => ChatMessage::tool_response(content()?, m.get("tool_call_id").and_then(Value::as_str).unwrap_or("")),
        "assistant" => {
            let reasoning = ["reasoning", "reasoning_content"]
                .iter()
                .find_map(|k| m.get(*k).and_then(Value::as_str))
                .filter(|r| !r.is_empty());
            let text = match m.get("content") {
                None | Some(Value::Null) => Vec::new(),
                Some(Value::String(s)) => vec![s.clone()],
                Some(Value::Array(parts)) => parts.iter().map(text_part).collect::<Result<_>>()?,
                Some(other) => bail!("assistant `content` is {other}"),
            };
            let calls = m.get("tool_calls").and_then(Value::as_array).map(Vec::as_slice).unwrap_or_default();
            let blocks: Vec<AssistantContentBlock> = reasoning
                .map(|r| AssistantContentBlock::Reasoning { text: r.to_string() })
                .into_iter()
                .chain(text.into_iter().map(|text| AssistantContentBlock::Text { text }))
                .chain(calls.iter().map(tool_call).collect::<Result<Vec<_>>>()?)
                .collect();
            if blocks.is_empty() {
                bail!("an assistant message has no text, reasoning or tool calls");
            }
            ChatMessage::assistant_blocks(blocks)
        }
        other => bail!("role `{other}` is not supported"),
    })
}

fn content(v: &Value) -> Result<ChatContent> {
    Ok(match v {
        Value::Null => ChatContent::Text(String::new()),
        Value::String(s) => ChatContent::Text(s.clone()),
        Value::Array(parts) => {
            ChatContent::Parts(parts.iter().map(|p| text_part(p).map(ChatContentPart::text)).collect::<Result<_>>()?)
        }
        other => bail!("`content` is {other}"),
    })
}

fn text_part(p: &Value) -> Result<String> {
    match (p.get("type").and_then(Value::as_str), p.get("text").and_then(Value::as_str)) {
        (Some("text"), Some(t)) => Ok(t.to_string()),
        _ => bail!("only text content parts are supported, got {p}"),
    }
}

fn tool(t: &Value) -> Result<ChatTool> {
    if t.get("type").and_then(Value::as_str).is_some_and(|k| k != "function") {
        bail!("only function tools are supported, got {t}");
    }
    let f = t.get("function").context("a tool has no `function`")?;
    let flag = |v: &Value| v.get("defer_loading").and_then(Value::as_bool);
    Ok(ChatTool {
        name: f.get("name").and_then(Value::as_str).context("a tool has no name")?.to_string(),
        description: f.get("description").and_then(Value::as_str).map(str::to_string),
        parameters: f.get("parameters").cloned().unwrap_or(Value::Null),
        strict: f.get("strict").and_then(Value::as_bool),
        defer_loading: flag(f).or(flag(t)),
    })
}

fn tool_call(c: &Value) -> Result<AssistantContentBlock> {
    if c.get("type").and_then(Value::as_str).is_some_and(|t| t != "function") {
        bail!("only function tool calls are supported, got {c}");
    }
    let f = c.get("function").context("a tool call has no `function`")?;
    Ok(AssistantContentBlock::ToolCall(AssistantToolCall {
        id: c.get("id").and_then(Value::as_str).unwrap_or("").to_string(),
        name: f.get("name").and_then(Value::as_str).context("a tool call has no name")?.to_string(),
        arguments: match f.get("arguments") {
            None | Some(Value::Null) => "{}".to_string(),
            Some(Value::String(s)) => s.clone(),
            Some(other) => other.to_string(),
        },
    }))
}
