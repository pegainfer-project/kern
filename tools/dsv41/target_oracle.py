"""Original forty-layer target oracle, directly bound to original checkpoint bytes.

Run with torchrun --nproc-per-node=4. The supplied Transformer.forward and all
its math remain unchanged; world_size uses its original TP+EP sharding. Only
construction excludes unused vision/draft modules, and the loader applies the
supplied convert.py slice rules in memory, without exporting any weights.
"""
import argparse
import ast
import contextlib
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch


def checkpoint_headers(checkpoint):
    index=json.loads((checkpoint/'model.safetensors.index.json').read_text())['weight_map']
    headers={}
    for shard in sorted(set(index.values())):
        with (checkpoint/shard).open('rb') as f:
            length=int.from_bytes(f.read(8),'little')
            headers.update({n:v for n,v in json.loads(f.read(length)).items() if n!='__metadata__'})
    if not set(index)<=headers.keys():raise ValueError('checkpoint tensor headers incomplete')
    return index,headers


def original_mapping(checkpoint):
    tree=ast.parse((checkpoint/'inference/convert.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='mapping' for t in n.targets))
    return ast.literal_eval(node.value)


def shard_spec(name,shape,rank,world,mapping):
    """Same partition precedence as original convert.py, expressed as a slice."""
    if '.experts.' in name:return None,0,shape[0],shape
    if '.engram.embed.' in name:
        count=(shape[0]+world-1)//world
        return 0,rank*count,count,[count,*shape[1:]]
    key=name.split('.')[-1] if any(x in name for x in ('hc','attn_sink','tie2eid','tid2eid','ape','image_')) else name.split('.')[-2]
    _,dim=mapping.get(key,(key,None))
    if dim is None:return None,0,shape[0],shape
    if shape[dim]%world:raise ValueError(f'{name}: indivisible shard')
    count=shape[dim]//world
    selected=list(shape);selected[dim]=count
    return dim,rank*count,count,selected


class TargetOracle:
    def __init__(self,checkpoint,*,batch,context,host_engram=False,inspect_world=None,inspect_rank=0):
        import torch
        from safetensors import safe_open
        from tokenizers import Tokenizer
        self.torch=torch;self.checkpoint=Path(checkpoint)
        self.index,self.headers=checkpoint_headers(self.checkpoint)
        self.mapping=original_mapping(self.checkpoint)
        self.files={};self.stack=contextlib.ExitStack();self.host_engram=host_engram
        sys.path.insert(0,str(self.checkpoint/'inference'))
        self.model=model=importlib.import_module('model')
        self.rank=int(os.environ.get('RANK','0'));self.world=int(os.environ.get('WORLD_SIZE','1'))
        if inspect_world is not None:self.rank,self.world=inspect_rank,inspect_world
        self.device=torch.device('meta' if inspect_world else f"cuda:{int(os.environ.get('LOCAL_RANK','0'))}")
        config=json.loads((self.checkpoint/'inference/config.json').read_text())
        config.update(max_batch_size=batch,max_seq_len=context,temperature=0,vision_n_layers=0,dspark_block_size=0)
        self.args=model.ModelArgs(**config)
        backend=Tokenizer.from_file(str(self.checkpoint/'tokenizer.json'))
        class TokenizerView:
            backend_tokenizer=backend
            def __len__(self):return backend.get_vocab_size(with_added_tokens=True)
        original_engram=model.ParallelEngramEmbedding
        class HostEngram(original_engram):
            def __init__(self,*args,**kwargs):
                with torch.device('cpu'):super().__init__(*args,**kwargs)
        torch.set_default_dtype(torch.bfloat16)
        with contextlib.ExitStack() as patches:
            if inspect_world:
                patches.enter_context(patch.object(model.dist,'is_initialized',return_value=True))
                patches.enter_context(patch.object(model.dist,'get_world_size',return_value=self.world))
                patches.enter_context(patch.object(model.dist,'get_rank',return_value=self.rank))
            elif host_engram:
                patches.enter_context(patch.object(model,'ParallelEngramEmbedding',HostEngram))
            with torch.device(self.device):self.target=model.Transformer(self.args,TokenizerView())
        self.plan=[]
        for name,param in self.target.named_parameters():
            if name not in self.index:raise ValueError(f'original reference parameter missing in checkpoint: {name}')
            header=self.headers[name]
            dim,start,count,shape=shard_spec(name,header['shape'],self.rank,self.world,self.mapping)
            if list(param.shape)!=shape:raise ValueError(f'{name}: reference shape {list(param.shape)} != source shard {shape}')
            self.plan.append({'name':name,'shape':shape,'dtype':str(param.dtype),'bytes':param.numel()*param.element_size(),
                              'host':host_engram and '.engram.embed.' in name})
        self.report={'rank':self.rank,'world_size':self.world,'parallelism':'original TP+EP','target_layers':len(self.target.layers),
                     'parameters':len(self.plan),'gpu_weight_bytes':sum(p['bytes'] for p in self.plan if not p['host']),
                     'host_weight_bytes':sum(p['bytes'] for p in self.plan if p['host']),
                     'buffer_bytes':sum(t.numel()*t.element_size() for t in self.target.buffers()),
                     'config':config,
                     'reference_sha256':{f:hashlib.sha256((self.checkpoint/'inference'/f).read_bytes()).hexdigest() for f in ('model.py','kernel.py','engram.py','convert.py')}}
        if inspect_world:return
        if self.rank == 0:print(json.dumps({'construction':self.report}),flush=True)
        for parameter_index,(name,param) in enumerate(self.target.named_parameters()):
            if parameter_index % 600 == 0:
                print(json.dumps({"rank":self.rank,"loading_parameter":parameter_index,"name":name}),flush=True)
            source=self.source(name)
            if name.endswith('wo_a.weight'):
                scale=self.source(name.removesuffix('weight')+'scale')
                # Exactly convert.py's BF16 O-A conversion, on this rank's slice.
                block_out=source.shape[0]//scale.shape[0];block_in=source.shape[1]//scale.shape[1]
                source=(source.unflatten(0,(-1,block_out)).unflatten(-1,(-1,block_in)).float()
                        *scale[:,None,:,None].float()).flatten(2,3).flatten(0,1).bfloat16()
            if param.dtype==torch.float4_e2m1fn_x2:source=source.view(param.dtype)
            with torch.no_grad():param.copy_(source)
        self.files.clear();self.stack.close()
        self.target.eval()
        torch.cuda.synchronize(self.device)
        self.report['allocated_after_load']=torch.cuda.memory_allocated(self.device)

    def source(self,name):
        import torch
        from safetensors import safe_open
        shard=self.index[name]
        if shard not in self.files:
            self.files[shard]=self.stack.enter_context(safe_open(self.checkpoint/shard,framework='pt',device='cpu'))
        with torch.device('cpu'):source=self.files[shard].get_tensor(name)
        dim,start,count,_=shard_spec(name,list(source.shape),self.rank,self.world,self.mapping)
        if dim is not None:
            actual=min(count,source.shape[dim]-start)
            source=source.narrow(dim,start,actual)
            if actual<count:
                shape=list(source.shape);shape[dim]=count-actual
                # Last Engram shard gets the same padding as original convert.py.
                padding=torch.full(shape,1 if name.endswith('.scale') else 0,dtype=torch.float32,device='cpu').to(source.dtype)
                source=torch.cat((source,padding),dim=dim)
        return source

    def forward(self,tokens,position):
        torch=self.torch
        embedding=self.model.F.embedding
        def staged_embedding(indices,weight,*args,**kwargs):
            if weight.device.type=='cpu' and indices.device.type=='cuda':
                # Only device transfers surround the original embedding operation;
                # the original Engram forward still performs its dequant/all_reduce.
                return embedding(indices.cpu(),weight,*args,**kwargs).to(indices.device)
            return embedding(indices,weight,*args,**kwargs)
        with torch.inference_mode(),torch.device(self.device),self.model.set_dtype(torch.bfloat16):
            with patch.object(self.model.F,'embedding',staged_embedding) if self.host_engram else contextlib.nullcontext():
                return self.target(tokens.to(device=self.device,dtype=torch.int64),position)

    def close(self):
        self.target=None;self.files.clear();self.stack.close()


def read_tokens(path):
    value=json.loads(Path(path).read_text())
    if isinstance(value,dict):value=value['tokens']
    if value and isinstance(value[0],int):value=[value]
    if not value or len({len(row) for row in value})!=1:raise ValueError('oracle batch requires equal-length token histories')
    return value


@contextlib.contextmanager
def capture_prefill(oracle, output):
    """Observe original module outputs; never replace inputs or returned tensors."""
    if output is None or (oracle.rank != 0 and not getattr(oracle, 'capture_all_ranks', False)):
        yield
        return
    if oracle.rank != 0:
        output = output / f'rank{oracle.rank}'
    output.mkdir(parents=True, exist_ok=True)
    entries = {}
    handles = []
    patches = contextlib.ExitStack()

    def save(name, value):
        value = value.detach().cpu().contiguous()
        dtype = str(value.dtype).removeprefix('torch.')
        suffix = {'bfloat16': 'bf16', 'float32': 'f32'}.get(dtype, 'bin')
        filename = f'{name}.{suffix}'
        (output / filename).write_bytes(value.view(oracle.torch.uint8).numpy().tobytes())
        entries[name] = {'file': filename, 'dtype': dtype, 'shape': list(value.shape)}

    def observe(name):
        def hook(module, args, value):
            save(name, value)
        return hook

    handles.append(oracle.target.embed.register_forward_hook(observe('embedding')))
    for index, layer in enumerate(oracle.target.layers):
        for name in ('attn_norm', 'attn', 'ffn_norm', 'ffn'):
            handles.append(getattr(layer, name).register_forward_hook(observe(f'layer.{index:02d}.{name}')))
        def block_hook(module, args, value, index=index):
            save(f'layer.{index:02d}.h', value[0])
            save(f'layer.{index:02d}.pre_mix', value[1])
        handles.append(layer.register_forward_hook(block_hook))
    diagnostic_layer = getattr(oracle, 'diagnostic_layer', None)
    if diagnostic_layer is not None:
        prefix = f'layer.{diagnostic_layer:02d}'
        attention = oracle.target.layers[diagnostic_layer].attn
        active = [False]
        def enter(module, args):
            active[0] = True
        def leave(module, args, value):
            active[0] = False
        handles.append(attention.register_forward_pre_hook(enter))
        handles.append(attention.register_forward_hook(leave))
        for name in ('wq_a', 'q_norm', 'wq_b', 'wkv', 'kv_norm', 'wo_b'):
            handles.append(getattr(attention, name).register_forward_hook(observe(prefix + '.' + name)))
        def before_wo_b(module, args):
            save(prefix + '.oa_output', args[0])
        handles.append(attention.wo_b.register_forward_pre_hook(before_wo_b))
        sparse = oracle.model.sparse_attn
        def traced_sparse(q, kv, sink, indices, scale):
            if active[0]:
                for name, value in (('q', q), ('kv', kv), ('sink', sink), ('indices', indices)):
                    save(prefix + '.sparse.' + name, value)
            result = sparse(q, kv, sink, indices, scale)
            if active[0]:
                save(prefix + '.sparse.output', result)
            return result
        patches.enter_context(patch.object(oracle.model, 'sparse_attn', traced_sparse))
        linear = oracle.model.linear
        def traced_linear(x, weight, bias=None):
            result = linear(x, weight, bias)
            if active[0] and weight is attention.wo_b.weight:
                save(prefix + '.wo_b_local', result)
            return result
        patches.enter_context(patch.object(oracle.model, 'linear', traced_linear))
    try:
        yield
    finally:
        patches.close()
        for handle in handles:
            handle.remove()
        (output / 'tensors.json').write_text(json.dumps(entries, indent=2) + '\n')


def trajectory(oracle,prompt,steps,output,teacher=None,prefill_hooks=None):
    torch=oracle.torch;output.mkdir(parents=True,exist_ok=True)
    batch,length=len(prompt),len(prompt[0]);history=torch.tensor(prompt,dtype=torch.int64,device=oracle.device)
    predictions=[];positions=[];elapsed=[]
    stream=(output/'target_logits.f32').open('wb') if oracle.rank==0 else contextlib.nullcontext()
    with stream as logits_file:
        for step in range(steps):
            position=0 if step==0 else length+step-1
            inputs=history if step==0 else history[:,-1:]
            started=time.monotonic()
            with capture_prefill(oracle,prefill_hooks if step == 0 else None):
                ids,logits,_=oracle.forward(inputs,position)
            torch.cuda.synchronize(oracle.device);elapsed.append(time.monotonic()-started)
            predictions.append(ids.cpu().tolist());positions.append(length-1+step)
            if oracle.rank==0:logits_file.write(logits.float().cpu().contiguous().numpy().tobytes())
            chosen=torch.tensor([row[step] for row in teacher],device=oracle.device,dtype=torch.int64) if teacher is not None and step<len(teacher[0]) else ids
            history=torch.cat((history,chosen[:,None]),dim=1)
            if oracle.rank==0:print(json.dumps({'step':step,'input_position':positions[-1],'ids':predictions[-1],'seconds':elapsed[-1]}),flush=True)
    if oracle.rank==0:
        report=dict(oracle.report,prompt_length=length,batch=batch,logits_shape=[steps,batch,oracle.args.vocab_size],
                    logits_first_input_position=length-1,input_positions=positions,predictions=predictions,
                    teacher_forced=teacher is not None,step_seconds=elapsed)
        (output/'trace.json').write_text(json.dumps(report,indent=2)+'\n')
        (output/'tokens.json').write_text(json.dumps(history.cpu().tolist())+'\n')


def diagnose_prefill(oracle, prompt, output):
    """Observe repeated original calls, then explicitly reset mutable caches."""
    phases = [('pure1', 1), ('pure2', 1)]
    if not getattr(oracle, 'diagnostic_pure_only', False):
        phases += [('with_decode', 4), ('after_decode', 1)]
    for phase, steps in phases:
        trajectory(oracle, prompt, steps, output / phase, prefill_hooks=output / phase / 'prefill-hooks')
    if getattr(oracle, 'diagnostic_pure_only', False):
        return
    reset = []
    with oracle.torch.no_grad():
        for name, value in oracle.target.named_buffers():
            suffix = name.rsplit('.', 1)[-1]
            if suffix in ('window_kv_cache', 'compress_kv_cache', 'k_cache', 'kv_state'):
                value.zero_(); reset.append(name)
            elif suffix == 'score_state':
                value.fill_(-float('inf')); reset.append(name)
            elif name == 'engram_hash.cache':
                value.fill_(-1); reset.append(name)
    oracle.model.shared_attn = oracle.model.SharedAttentionRuntime()
    trajectory(oracle, prompt, 1, output / 'after_reset', prefill_hooks=output / 'after_reset' / 'prefill-hooks')
    if oracle.rank == 0:
        (output / 'reset.json').write_text(json.dumps({'buffers': reset, 'shared_attn': 'fresh original SharedAttentionRuntime'}, indent=2) + '\n')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--tokens',type=Path);p.add_argument('--teacher-tokens',type=Path)
    p.add_argument('--steps',type=int,default=8);p.add_argument('--ar-steps',type=int,default=0)
    p.add_argument('--context',type=int,default=4096);p.add_argument('--batch',type=int,default=1)
    p.add_argument('--host-engram',action='store_true');p.add_argument('--inspect-world',type=int)
    p.add_argument('--inspect-rank',type=int,default=0)
    p.add_argument('--diagnostic-pure-only',action='store_true')
    p.add_argument('--diagnostic-layer',type=int,help='Capture internal attention boundaries in this layer')
    p.add_argument('--capture-all-ranks',action='store_true')
    p.add_argument('--diagnose-prefill',action='store_true',help='Repeated prefill, decode/prefill and cache-reset diagnostic')
    p.add_argument('--prefill-hooks',action='store_true',help='Save original first-prefill module outputs without changing computation')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    import torch
    if a.inspect_world:
        oracle=TargetOracle(a.checkpoint,batch=a.batch,context=a.context,host_engram=a.host_engram,inspect_world=a.inspect_world,inspect_rank=a.inspect_rank)
        (a.out/'memory.json').write_text(json.dumps(oracle.report,indent=2)+'\n')
        (a.out/'parameter-plan.json').write_text(json.dumps(oracle.plan,indent=2)+'\n')
        print(json.dumps(oracle.report,indent=2));return
    if not a.tokens:p.error('--tokens is required for execution')
    prompt=read_tokens(a.tokens);teacher=read_tokens(a.teacher_tokens) if a.teacher_tokens else None
    if teacher and len(teacher)!=len(prompt):raise ValueError('teacher batch mismatch')
    if teacher and len(teacher[0])<a.steps-1:raise ValueError('teacher must provide each subsequent input token')
    if len(prompt[0])+max(a.steps,a.ar_steps)>a.context:raise ValueError('context bound too small')
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK','0')))
    if int(os.environ.get('WORLD_SIZE','1'))>1:torch.distributed.init_process_group('nccl')
    torch.set_num_threads(4)
    oracle=TargetOracle(a.checkpoint,batch=len(prompt),context=a.context,host_engram=a.host_engram)
    oracle.diagnostic_pure_only = a.diagnostic_pure_only
    oracle.diagnostic_layer = a.diagnostic_layer
    oracle.capture_all_ranks = a.capture_all_ranks
    try:
        if a.diagnose_prefill:
            diagnose_prefill(oracle,prompt,a.out)
            return
        trajectory(oracle,prompt,a.steps,a.out/'teacher' if teacher else a.out/'ar',teacher,
                   a.out/'prefill-hooks' if a.prefill_hooks else None)
        if teacher and a.ar_steps:trajectory(oracle,prompt,a.ar_steps,a.out/'ar')
    finally:
        oracle.close()
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()


if __name__=='__main__':main()
