#!/usr/bin/env python3
"""Prepare CPU-only canonical trace configs with the checkpoint's chat encoder."""
import argparse
import importlib.util
import json
from pathlib import Path

PROMPTS = {
    'english-science': 'Explain why leaves change color in autumn in three clear sentences.',
    'chinese-explanation': '请用三段简洁的中文说明城市为什么会出现热岛效应，并给出两项适合居民参与的改善措施。',
    'python-code': 'Write a Python function that returns the longest substring without repeated characters. Explain the sliding-window invariant and give two small tests.',
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint', 'manifest', 'kernels', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--generation-tokens', type=int, default=96)
    args = p.parse_args()
    from transformers import AutoTokenizer
    spec = importlib.util.spec_from_file_location('checkpoint_encoding', args.checkpoint / 'encoding/encoding.py')
    encoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoder)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    for name, prompt in PROMPTS.items():
        directory = args.output / name
        directory.mkdir(parents=True, exist_ok=True)
        messages = [{'role': 'user', 'content': prompt}]
        text = encoder.encode_messages(messages, thinking_mode='chat')
        ids = tokenizer.encode(text, add_special_tokens=False)
        config = dict(manifest=str(args.manifest), kernels=str(args.kernels),
                      weights=[[str(args.checkpoint)]] * 4, prompt_ids=ids,
                      generation_tokens=args.generation_tokens, capacity_tokens=32768,
                      draft_proposals=True, target_logits=True, output=str(directory))
        (directory / 'config.json').write_text(json.dumps(config, indent=2))
        (directory / 'request.json').write_text(json.dumps(
            {'messages': messages, 'temperature': 0, 'thinking_mode': 'chat', 'rendered_prompt': text},
            ensure_ascii=False, indent=2))
        print(name, len(ids), 'prompt tokens')


if __name__ == '__main__':
    main()
