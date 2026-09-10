"""Create eight same-history pairs from a canonical greedy AR trace."""
import argparse,json,pathlib


def main():
    p=argparse.ArgumentParser();p.add_argument('trace_config',type=pathlib.Path);p.add_argument('output',type=pathlib.Path);p.add_argument('--positions',type=int,nargs='+');a=p.parse_args()
    config=json.loads(a.trace_config.read_text());trace=pathlib.Path(config['output']);meta=json.loads((trace/'trace.json').read_text());ids=json.loads((trace/'tokens.json').read_text())
    starts=a.positions or [meta['prompt_length']+i*6 for i in range(8)]
    if len(starts)!=8:raise ValueError('Use eight positions for the c32 geometry')
    if any(p<1 or p+6>len(ids) for p in starts):raise ValueError('Trace does not cover all six teacher inputs')
    config.update(output=str(a.output.resolve()),histories=[[ids[:p] for p in starts] for _ in range(4)],teacher_ids=[[ids[p:p+6] for p in starts] for _ in range(4)])
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    print(a.output/'config.json')

if __name__=='__main__':main()
