#!/usr/bin/env python3
"""Original DSpark acceptance conditioned on a canonical kern AR target trace.

This is a draft numerical/acceptance oracle, not an independent target oracle.
Every AR position advances the original draft caches, including positions a
speculative scheduler would skip. Round statistics then follow prefix acceptance.
"""
import argparse
import json
from pathlib import Path
import sys
import time


def summarize(records, first_anchor):
    by_anchor = {r['anchor_position']: r for r in records}
    rounds = []
    anchor = first_anchor
    while anchor in by_anchor:
        row = by_anchor[anchor]
        rounds.append(row)
        anchor += row['accepted_length']
    return {
        'rounds': len(rounds),
        'mean_accepted_length': sum(r['accepted_length'] for r in rounds) / len(rounds) if rounds else None,
        'accepted_length_histogram': {str(k): sum(r['accepted_length'] == k for r in rounds) for k in range(1, 7)},
        'conditional_prefix_survival': [sum(r['draft_prefix'] >= k for r in rounds) / len(rounds) if rounds else None for k in range(1, 6)],
        'selected_rounds': rounds,
    }


def proposal_differences(reference, actual):
    differing = [0] * 5
    first = [0] * 5
    delta = {}
    for r, k in zip(reference, actual):
        bad = [i for i, (x, y) in enumerate(zip(r['draft_ids'], k['draft_ids'])) if x != y]
        for i in bad:
            differing[i] += 1
        if bad:
            first[bad[0]] += 1
        d = str(k['draft_prefix'] - r['draft_prefix'])
        delta[d] = delta.get(d, 0) + 1
    return {'differing_proposals_by_depth': differing, 'first_differing_depth_counts': first,
            'kern_minus_reference_prefix_histogram': delta,
            'matching_target_prefix_equal': delta.get('0', 0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--eos-id', type=int, action='append', default=[])
    parser.add_argument('--max-positions', type=int)
    parser.add_argument('--kern-proposals', type=Path,
                        help='kern_proposals.json from the same canonical trace')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from dsv41.moe.dspark_oracle import DraftOracle

    meta = json.loads((args.trace / 'trace.json').read_text())
    if meta.get('teacher_forced'):
        raise ValueError('Acceptance requires a real greedy AR target history, not a teacher-forced trace')
    ids = json.loads((args.trace / meta['tokens']).read_text())
    prompt = meta['prompt_length']
    assert meta['tap_dtype'] == 'bf16' and meta['tap_width'] == 15360
    taps = torch.frombuffer(bytearray((args.trace / meta['taps']).read_bytes()), dtype=torch.bfloat16)
    taps = taps.reshape(meta['tap_rows'], 15360)
    eos_ids = args.eos_id
    if not eos_ids:
        from transformers import AutoTokenizer
        eos = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True).eos_token_id
        eos_ids = [eos] if isinstance(eos, int) else list(eos or [])
    stop = next((p for p in range(prompt, len(ids)) if ids[p] in eos_ids), len(ids))
    # Only score rounds with all five target comparison tokens available before EOS.
    anchors = range(prompt, min(stop - 5, meta['tap_rows'] + 1))
    if args.max_positions is not None:
        anchors = range(anchors.start, min(anchors.stop, anchors.start + args.max_positions))
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    oracle = DraftOracle(args.checkpoint, max_seq_len=max(256, len(ids) + 8), device=args.device)
    records = []
    kern_records = []
    kern_proposals = ({r['anchor_position']: r for r in json.loads(args.kern_proposals.read_text())}
                      if args.kern_proposals else {})
    with torch.inference_mode():
        # Seed all context preceding the first anchor. The next call re-writes
        # its final row exactly, as the original single-row decode interface does.
        oracle.draft(torch.tensor([ids[prompt]], device=args.device), taps[:prompt].unsqueeze(0).to(args.device), 0)
        for anchor in anchors:
            prediction, logits, confidence = oracle.draft(
                torch.tensor([ids[anchor]], device=args.device),
                taps[anchor - 1:anchor].unsqueeze(0).to(args.device), anchor - 1)
            proposal = prediction[0].tolist()
            assert proposal[0] == ids[anchor], 'reference output must include unchanged anchor'
            proposal = proposal[1:6]
            target = ids[anchor + 1:anchor + 6]
            prefix = 0
            for a, b in zip(proposal, target):
                if a != b:
                    break
                prefix += 1
            row = {'anchor_position': anchor, 'anchor_id': ids[anchor], 'draft_ids': proposal,
                   'target_ids': target, 'draft_prefix': prefix, 'accepted_length': 1 + prefix,
                   'confidence': confidence[0].float().tolist()}
            records.append(row)
            if kern_proposals:
                actual = kern_proposals[anchor]
                assert actual['anchor_id'] == ids[anchor], 'kern proposal history differs'
                kp = 0
                for x, y in zip(actual['draft_ids'], target):
                    if x != y:
                        break
                    kp += 1
                kern_records.append({**actual, 'target_ids': target, 'draft_prefix': kp, 'accepted_length': 1 + kp})
                row['kern_draft_ids'] = actual['draft_ids']
                row['draft_ids_equal'] = proposal == actual['draft_ids']
            print(json.dumps(row), flush=True)
            # Preserve incremental evidence if a later original-kernel JIT fails.
            with (args.output / 'positions.jsonl').open('a' if len(records) > 1 else 'w') as out:
                out.write(json.dumps(row) + '\n')
    report = {'scope': 'Original checkpoint DSpark conditioned on kern greedy target taps and token IDs',
              'independent_target_oracle': False, 'proposal_length': 5, 'temperature': 0,
              'prompt_length': prompt, 'eos_position': stop if stop < len(ids) else None,
              'tail_policy': 'Only anchors with five following target tokens strictly before EOS',
              'evaluated_positions': len(records), 'elapsed_seconds_including_load_and_jit': time.monotonic() - started,
              **summarize(records, prompt)}
    if kern_records:
        report['kern_on_same_history'] = summarize(kern_records, prompt)
        report['proposal_exact_positions'] = sum(r['draft_ids_equal'] for r in records)
        report['proposal_differences'] = proposal_differences(records, kern_records)
    (args.output / 'acceptance.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'selected_rounds'}, indent=2))
    oracle.close()


if __name__ == '__main__':
    main()
