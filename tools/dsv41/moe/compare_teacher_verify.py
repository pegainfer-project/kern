"""Compare same-history full-vocabulary plain and verify logits."""
import argparse,json,pathlib
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('case',type=pathlib.Path);a=p.parse_args();report=[]
    for rank in range(4):
        d=a.case/f'rank{rank}';m=json.loads((d/'metadata.json').read_text());shape=(m['seqs'],m['width'],m['vocab'])
        x=np.fromfile(d/'plain.f32',dtype='<f4').reshape(shape);y=np.fromfile(d/'verify.f32',dtype='<f4').reshape(shape)
        if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError(f'rank{rank}: nonfinite logits')
        committed=np.fromfile(d/'after_commit.f32',dtype='<f4').reshape(shape[0],shape[2])
        if not np.isfinite(committed).all():raise ValueError(f'rank{rank}: nonfinite committed logits')
        for seq in range(shape[0]):
            for phase,step in [('verify',i) for i in range(shape[1])]+[('after_commit',m['accepted'])]:
                u=x[seq,step].astype(np.float64)
                v=(y[seq,step] if phase=='verify' else committed[seq]).astype(np.float64)
                ui=np.argsort(u)[-5:][::-1];vi=np.argsort(v)[-5:][::-1];delta=v-u
                report.append(dict(phase=phase,rank=rank,sequence=seq,step=step,position=m['positions'][seq]+step,
                    plain_top5=ui.tolist(),verify_top5=vi.tolist(),plain_scores=u[ui].tolist(),verify_scores=v[vi].tolist(),
                    top1_equal=bool(ui[0]==vi[0]),plain_margin=float(u[ui[0]]-u[ui[1]]),verify_margin=float(v[vi[0]]-v[vi[1]]),
                    max_abs=float(np.abs(delta).max()),rmse=float(np.sqrt(np.mean(delta**2))),
                    relative_squared=float(np.sum(delta**2)/(np.sum(u**2)+1e-30)),
                    plain_winner_delta=float(delta[ui[0]]),verify_winner_delta=float(delta[vi[0]])))
    result={'rows':len(report),'top1_mismatches':sum(not r['top1_equal'] for r in report),
            'max_abs':max(r['max_abs'] for r in report),'max_relative_squared':max(r['relative_squared'] for r in report),'details':report}
    (a.case/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='details'},indent=2))
    for r in report:
        if not r['top1_equal']:print(json.dumps(r))

if __name__=='__main__':main()
