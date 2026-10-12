//! Native rendering is CPU-only and does not require a checkpoint chat template.
use vllm_chat::{ChatMessage, ChatRenderer, ChatRequest, DeepSeekV4ChatRenderer};

#[test]
fn native_nonthinking_user_prompt_matches_supplied_v41_encoding() {
    let mut request = ChatRequest { messages: vec![ChatMessage::user("Hello.")], ..ChatRequest::for_test() };
    request.chat_options.template_kwargs.insert("thinking".into(), serde_json::Value::Bool(false));
    let actual = DeepSeekV4ChatRenderer::new(Default::default()).render(&request).unwrap().prompt.into_text().unwrap();
    // Generated with the supplied encoding.py encode_messages(..., thinking_mode="chat").
    // This deliberately covers only the common nonthinking text format; V4.1
    // changes system messages, reasoning effort and tool tags.
    assert_eq!(actual, "<｜begin▁of▁sentence｜><｜User｜>Hello.<｜Assistant｜></think>");
}

#[test]
fn v41_matches_released_reference_text_cases() {
    use serde_json::Value;
    use vllm_chat::{AssistantContentBlock, AssistantToolCall, ChatTool, DeepSeekV41ChatRenderer, ResolvedToolContext};
    let cases: Vec<Value> = serde_json::from_str(include_str!("fixtures/deepseek_v41.json")).unwrap();
    for case in cases {
        let mut messages = Vec::new();
        let mut tools: Vec<ChatTool> = Vec::new();
        for message in case["messages"].as_array().unwrap() {
            if let Some(list) = message["tools"].as_array() {
                tools.extend(
                    list.iter().map(|tool| serde_json::from_value::<ChatTool>(tool["function"].clone()).unwrap()),
                );
            }
            let text = message["content"].as_str().unwrap_or("");
            messages.push(match message["role"].as_str().unwrap() {
                "system" => ChatMessage::system(text),
                "user" => ChatMessage::user(text),
                "assistant" => {
                    let mut blocks = Vec::new();
                    if let Some(reasoning) = message["reasoning_content"].as_str() {
                        blocks.push(AssistantContentBlock::Reasoning { text: reasoning.into() });
                    }
                    if !text.is_empty() {
                        blocks.push(AssistantContentBlock::Text { text: text.into() });
                    }
                    if let Some(calls) = message["tool_calls"].as_array() {
                        for call in calls {
                            let f = &call["function"];
                            blocks.push(AssistantContentBlock::ToolCall(AssistantToolCall {
                                id: call["id"].as_str().or(f["id"].as_str()).unwrap_or("").into(),
                                name: f["name"].as_str().unwrap().into(),
                                arguments: f["arguments"]
                                    .as_str()
                                    .map(str::to_owned)
                                    .unwrap_or_else(|| f["arguments"].to_string()),
                            }));
                        }
                    }
                    ChatMessage::assistant_blocks(blocks)
                }
                "tool" => ChatMessage::tool_response(text, message["tool_call_id"].as_str().unwrap_or("")),
                other => panic!("unsupported fixture role {other}"),
            });
        }
        let tool_context = ResolvedToolContext::new(&messages, tools, None, true).unwrap();
        let mut request = ChatRequest { messages, tool_context, ..ChatRequest::for_test() };
        for (key, source) in
            [("thinking", "thinking"), ("reasoning_effort", "effort"), ("drop_thinking", "drop_thinking")]
        {
            request.chat_options.template_kwargs.insert(key.into(), case[source].clone());
        }
        let rendered = DeepSeekV41ChatRenderer::new(Default::default()).render(&request);
        if case["error"] == Value::Bool(true) {
            assert!(rendered.is_err(), "{} should fail", case["name"]);
        } else {
            // The checkpoint oracle decodes double-encoded objects twice.
            // deepseek-recipe (and vLLM #56260) instead preserves that raw
            // string as `arguments`; keep the original fixture intact and
            // require the rest of the rendered conversation to match it.
            let expected = case["expected"].as_str().unwrap().replace(
                r#"<｜DSML｜ parameter name="query" string="true">double</｜DSML｜ parameter>"#,
                r#"<｜DSML｜ parameter name="arguments" string="true">"{\"query\":\"double\"}"</｜DSML｜ parameter>"#,
            );
            assert_eq!(rendered.unwrap().prompt.into_text().unwrap(), expected, "{}", case["name"]);
        }
    }
}

#[test]
fn v41_typed_reasoning_effort_controls_numeric_budget() {
    use vllm_chat::{DeepSeekV41ChatRenderer, EffortValue};
    let render = |request: &ChatRequest| DeepSeekV41ChatRenderer::new(Default::default()).render(request);
    for (effort, budget) in [("low", 50), ("high", 75), ("xhigh", 75), ("max", 100)] {
        let mut request = ChatRequest::for_test();
        request.chat_options.reasoning_effort = Some(EffortValue::from(effort));
        let prompt = render(&request).unwrap().prompt.into_text().unwrap();
        assert!(
            prompt.starts_with(&format!("<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: {budget} (range 1-100,"))
        );
    }
    for effort in ["minimal", "medium"] {
        let mut request = ChatRequest::for_test();
        request.chat_options.reasoning_effort = Some(EffortValue::from(effort));
        assert!(render(&request).is_err(), "{effort}");
    }
}

/// The PP head checkpoints a recurrent line at `history_tokens`: the prompt
/// minus the generation prompt tail, which pegainfer measures on the last
/// message alone. For Kimi K3 that is sound only while the conversation
/// rendered without a generation prompt is a prefix of the next turn's prompt
/// and the tail the whole conversation adds equals what the last message alone
/// adds. Checked over every user/assistant shape up to four messages that ends
/// in a user turn, the shape AgentX replays.
#[test]
fn k3_history_is_a_prefix_of_the_next_turn_and_the_tail_is_local() {
    use std::sync::Arc;
    use vllm_chat::{GenerationPromptMode, KimiK3ChatRenderer};
    use vllm_tokenizer::test_utils::TestTokenizer;

    let tokenizer = ["<|open|>", "<|close|>", "<|sep|>", "<|end_of_msg|>"]
        .into_iter()
        .zip(256..)
        .fold(TestTokenizer::new(), |t, (token, id)| t.with_special_token(token, id));
    let k3 = KimiK3ChatRenderer::new(Arc::new(tokenizer), Default::default());
    let render = |messages: &[ChatMessage], mode| {
        let mut request = ChatRequest { messages: messages.to_vec(), ..ChatRequest::for_test() };
        request.chat_options.generation_prompt_mode = mode;
        k3.render(&request).unwrap().prompt.into_token_ids().unwrap()
    };
    let tail = |messages: &[ChatMessage]| {
        render(messages, GenerationPromptMode::StartNewAssistant).len()
            - render(messages, GenerationPromptMode::NoGenerationPrompt).len()
    };
    let texts = ["Hello.", "", " padded\n", "literal <|open|>message<|sep|>"];
    let message =
        |user: bool, text: &str| if user { ChatMessage::user(text) } else { ChatMessage::assistant_text(text) };
    let mut level: Vec<Vec<ChatMessage>> = vec![vec![]];
    let mut shapes = level.clone();
    for _ in 0..3 {
        level = level
            .iter()
            .flat_map(|s| {
                texts.iter().flat_map(move |t| [true, false].map(|u| [s.clone(), vec![message(u, t)]].concat()))
            })
            .collect();
        shapes.extend(level.iter().cloned());
    }
    let mut checked = 0;
    for history in &shapes {
        for text in texts {
            let turn = [history.clone(), vec![ChatMessage::user(text)]].concat();
            let next = [turn.clone(), vec![ChatMessage::assistant_text("reply"), ChatMessage::user("next")]].concat();
            let prompt = render(&turn, GenerationPromptMode::StartNewAssistant);
            let boundary = prompt.len() - tail(&turn[turn.len() - 1..]);
            let next = render(&next, GenerationPromptMode::StartNewAssistant);
            assert_eq!(
                (prompt[..boundary] == next[..boundary], tail(&turn), boundary),
                (true, prompt.len() - boundary, render(&turn, GenerationPromptMode::NoGenerationPrompt).len()),
                "{turn:?}"
            );
            checked += 1;
        }
    }
    assert_eq!(checked, (1 + 8 + 64 + 512) * 4);
}
