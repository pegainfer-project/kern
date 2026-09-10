"""Locate first divergent checkpoint in teacher-forced plain/verify execution."""
import argparse,json,pathlib,struct


def values(data,dtype):
    if dtype=='bf16':return [struct.unpack('<f',struct.pack('<I',v[0]<<16))[0] for v in struct.iter_unpack('<H',data)]
    return [v[0] for v in struct.iter_unpack({'f32':'<f','i32':'<i','i64':'<q'}[dtype],data)]


def main():
    p=argparse.ArgumentParser();p.add_argument('rank',type=pathlib.Path);a=p.parse_args();plain=a.rank/'plain-boundary';verify=a.rank/'verify-boundary'
    xi=json.loads((plain/'index.json').read_text());yi={x['label']:x for x in json.loads((verify/'index.json').read_text())};report=[]
    for x in xi:
        if x['label'] not in yi:continue
        y=yi[x['label']];u=(plain/x['file']).read_bytes();v=(verify/y['file']).read_bytes();assert len(u)==len(v)
        row={'label':x['label'],'buffer':x['buffer'],'bytes':len(u),'equal':u==v}
        if u!=v:
            uf,vf=values(u,x['dtype']),values(v,x['dtype']);row.update(max_abs=max(abs(i-j) for i,j in zip(uf,vf)),relative_squared=sum((i-j)**2 for i,j in zip(uf,vf))/(sum(i*i for i in uf)+1e-30))
        report.append(row)
    result={'boundaries':len(report),'first_difference':next((r for r in report if not r['equal']),None),'details':report}
    (a.rank/'boundary-comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='details'},indent=2))
    for row in report:
        if not row['equal']:print(json.dumps(row))

if __name__=='__main__':main()
