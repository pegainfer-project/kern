"""Compare actual Runtime probe dumps to the supplied-inference fixtures."""
import argparse,json,pathlib,struct


def floats(data,dtype):
    if dtype=='f32':return [v[0] for v in struct.iter_unpack('<f',data)]
    return [struct.unpack('<f',struct.pack('<I',v[0]<<16))[0] for v in struct.iter_unpack('<H',data)]


def main():
    p=argparse.ArgumentParser();p.add_argument('case',type=pathlib.Path);a=p.parse_args();io=json.loads((a.case/'io.json').read_text());manifest=json.loads((a.case/'manifest.json').read_text());report={}
    for category in ('outputs','state_outputs'):
        for name,file in io.get(category,{}).items():
            expected=(a.case/file).read_bytes();actual=(a.case/(name+'.kern.bin')).read_bytes();assert len(actual)==len(expected)
            if name in io.get('compare_rows',{}):
                size=len(expected)//io['vars']['tokens']*io['compare_rows'][name];expected,actual=expected[:size],actual[:size]
            dtype=manifest['buffers'][name]['dtype'] if category=='outputs' else 'u8'
            entry={'bytes':len(expected),'mismatched_bytes':sum(a!=b for a,b in zip(actual,expected))}
            if dtype in ('bf16','f32'):
                actual_f,expected_f=floats(actual,dtype),floats(expected,dtype)
                relative=sum((x-y)**2 for x,y in zip(actual_f,expected_f))/(sum(y*y for y in expected_f)+1e-30)
                entry['relative_squared_error']=relative;assert relative<io.get('limits',{}).get(name,1e-5),(name,relative)
            else:assert actual==expected,name
            report[name]=entry
    for output,spec in io.get('padding_passthrough',{}).items():
        actual=(a.case/(output+'.kern.bin')).read_bytes();source=(a.case/io['inputs'][spec['input']]).read_bytes();start=len(source)//io['vars']['tokens']*spec['start']
        assert actual[start:]==source[start:],(output,'padding changed')
        report[output]['padding_byte_exact']=True
    (a.case/'comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
