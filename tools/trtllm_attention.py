"""TRTLLM-GEN Q24/KV4/D256 BF16 attention as a kern manifest op.

ABI observed from FlashInfer 0.6.16.post3 on SM103. No upstream source or
binary is vendored; artifacts.json pins NVIDIA-hosted cubins by SHA256.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import urllib.request

ARTIFACTS = Path(__file__).with_name('trtllm-gen') / 'artifacts.json'
PAGE, HEADS, KV_HEADS, HEAD_DIM = 64, 24, 4, 256
ROW_BYTES = HEADS * HEAD_DIM * 2
LAYER_PAGE_BYTES = PAGE * KV_HEADS * HEAD_DIM * 4


def artifacts():
    return json.loads(ARTIFACTS.read_text())['kernels']


def fetch(out):
    """Download only exact pinned files, validating existing files too."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for artifact in artifacts().values():
        dst = out / artifact['source']
        if dst.exists() and hashlib.sha256(dst.read_bytes()).hexdigest() == artifact['sha256']:
            continue
        with urllib.request.urlopen(artifact['url'], timeout=120) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != artifact['sha256']:
            raise ValueError(f"SHA256 mismatch for {artifact['source']}")
        tmp = dst.with_suffix('.download')
        tmp.write_bytes(data)
        tmp.replace(dst)


def op(mode, *, max_rows, max_seqs, max_context, layers=16, splits=38, kv_type='in state'):
    """Interface: out, q, k, v, page_table, seq_lens, cu_q, batch, rows.

    Context mode handles prefill / causal verification; decode consumes one
    query per sequence. KV pages may be layer-interleaved with arbitrary
    layer offsets supplied at the call site. Q and O are contiguous.
    Decode uses a fixed split count: graph topology never depends on host
    reads of GPU sequence lengths. Scratch scales with the declared batch
    bound, rather than assuming that CTA count never exceeds SM count.
    """
    if mode not in ('prefill', 'decode'):
        raise ValueError('mode must be prefill or decode')
    if not 1 <= splits <= 128:
        raise ValueError('splits must be between 1 and 128')
    if min(max_rows, max_seqs, max_context, layers) < 1:
        raise ValueError('all bounds must be positive')
    a = artifacts()['prefill' if mode == 'prefill' else 'decode' if splits == 1 else 'decode_split']
    table_width = (max_context + PAGE - 1) // PAGE
    tmap = lambda param, dims, strides, box: dict(param=param, dtype='bf16', dims=dims,
                                                  strides=strides, box=box, swizzle=128, l2_promotion=128)
    context = mode == 'prefill'
    qmap = tmap(1, [256, 1 if context else 6, 24 if context else 4, max_rows],
                [512, 512 if context else 3072, ROW_BYTES], [64, 1 if context else 6, 1, 128 if context else 1])
    kmap = lambda param: tmap(param, [256, PAGE, 4, 0], [4096, 1024, layers * LAYER_PAGE_BYTES], [64, 64, 1, 1])
    # Final dimension is a singleton: its unused upstream zero stride is
    # represented as 16 to satisfy kern's positive-stride contract.
    omap = tmap(0, [256, max_rows, 4, 6, 1], [ROW_BYTES, 512, 2048, 16], [64, 128 if context else 8, 1, 1, 1])
    fields = [dict(at=at, tensormap=t) for at,t in [(0,qmap),(128,kmap(2)),(384,kmap(3)),(512,omap)]]
    fields += [dict(at=at, param=p) for at,p in [(912,0),(1000,4),(1080,5),(1116,7),(1244,8)]]
    if context:
        fields.append(dict(at=936,param=6))
    constants = {1112:2147483647, 1144:1, 1148:max_context, 1152:1, 1156:1 if context else splits,
                 1160:table_width, 1164:4, 1168:24, 1172:6, 1176:6, 1180:-1431655765, 1188:2,
                 1192:6144, 1200:2*max_seqs*table_width, 1204:128 if context else 1,
                 1208:6, 1212:1, 1216:1065353216, 1220:1035512379,
                 1224:1065353216, 1228:-1082130432, 1252:256}
    if context:
        # Overlaunch across ragged sequences is legal: each CTA checks cu_q.
        del constants[1144],constants[1152]
        fields += [dict(at=1144,param=8),dict(at=1152,expr={'ceil_div':['tokens',128]})]
    fields += [dict(at=at,i32=value) for at,value in constants.items()]
    scratch = {}
    if not context and splits > 1:
        scratch = {'counter':dict(dtype='i32',shape=[max_seqs*HEADS]),
                   'stats':dict(dtype='f32',shape=[max_seqs,KV_HEADS,splits,8,2]),
                   'partial':dict(dtype='f32',shape=[max_seqs,KV_HEADS,splits,8,HEAD_DIM])}
        fields += [dict(at=at,scratch=n) for at,n in [(984,'counter'),(1008,'partial'),(1016,'stats')]]
    launch = dict(cubin=a['source'],sha256=a['sha256'],entry=a['entry'],block=a['block'],
                  grid=[{'ceil_div':['tokens',128]} if context else splits,24 if context else 4,'seqs'],
                  shared_mem=a['shared_mem'],params=['bytes<1280>'],args=[dict(pack=dict(size=1280,fields=fields))])
    return dict(params=['out buffer<bf16>','in buffer<bf16>',kv_type,kv_type,
                       'in buffer<i32>','in buffer<i32>','in buffer<i32>','i32','i32'],
                impl=dict(scratch=scratch,launches=[launch]))


def convert(manifest, cache, *, splits=38):
    """Replace target full attention and its page-64 KV append in a Qwen3.8 manifest."""
    from kern_manifest import normalize, resolve_constants
    m = copy.deepcopy(resolve_constants(manifest))
    if m['model'] not in ('qwen3.8-27b', 'qwen3.8-27b-dflash2'):
        raise ValueError('expected an unconverted Qwen3.8-27B manifest')
    if m['buffers']['q_n']['shape'][-1] != HEADS * HEAD_DIM:
        raise ValueError('expected Q24/KV4/D256 BF16 full attention')
    old_page=m['buffers']['block_table']['domain']['stride']
    old_layer_bytes=old_page*4096
    old_block_stride=old_layer_bytes*16//2
    max_context=262144
    width=(max_context+PAGE-1)//PAGE
    m['buffers']['block_table']['shape'][-1]=width
    m['buffers']['block_table']['domain']['stride']=PAGE
    max_rows=m['vars']['tokens']['max'];max_seqs=m['vars']['seqs']['max']
    wanted={'attn','attn_batch','attn_prefill','attn_verify'} & m['ops'].keys()
    for name in wanted:
        m['ops'][name]=op('decode' if name in ('attn','attn_batch') else 'prefill',
                          max_rows=max_rows,max_seqs=max_seqs,max_context=max_context,splits=splits)
    for p in m['programs'].values():
        for call in p['calls']:
            if call['op'] in wanted:
                # The normalized old interface's first six buffers are stable;
                # locate cu_q by identity instead of relying on folded scalar positions.
                cuq=next(a for a in call['args'] if a.get('buf')=='cu_seqlens_q')
                call['args']=call['args'][:6]+[cuq,{'var':'seqs'},{'var':'tokens'}]
            for arg in call['args']:
                if arg.get('state')=='kv':
                    layer,within=divmod(arg.get('offset',0),old_layer_bytes)
                    arg['offset']=layer*LAYER_PAGE_BYTES+within
    launch=m['ops']['reshape_and_cache']['impl']['launches'][0]
    launch.pop('module',None)
    launch.update(cubin=cache['source'],sha256=cache['sha256'],entry=cache['entry'])
    for arg in launch.get('args',[]):
        if arg.get('i64')==old_block_stride:arg['i64']=16*LAYER_PAGE_BYTES//2
    m['model']+='-trtllm-gen'
    # Remove replaced modules (normalize hoists the new artifacts below).
    used={l['module'] for o in m['ops'].values() for l in o['impl']['launches'] if 'module' in l}
    m['modules']={k:v for k,v in m['modules'].items() if k in used}
    return normalize(m)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--download',type=Path)
    p.add_argument('--input',type=Path)
    p.add_argument('--cache-metadata',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--splits',type=int,default=38)
    args=p.parse_args()
    if args.download:fetch(args.download)
    if args.input:
        if not args.cache_metadata or not args.output:p.error('--input requires --cache-metadata and --output')
        result=convert(json.loads(args.input.read_text()),json.loads(args.cache_metadata.read_text()),splits=args.splits)
        args.output.write_text(json.dumps(result,indent=1)+'\n')

if __name__=='__main__':main()
