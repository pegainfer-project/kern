"""Load-time weight preparation interfaces (call from `once`)."""
from pathlib import Path
import hashlib

def pieces(n=2304,k=5120,row_group=1,gate_up=True,cubin_dir=None):
 root=Path(cubin_dir) if cubin_dir else Path(__file__).resolve().parents[3]/'target/cubins/dsv41'
 modules={'dsv41_weight_prep':{'source':str((Path(cubin_dir) if cubin_dir else Path('target/cubins/dsv41'))/'weight_prep.cubin'),'sha256':hashlib.sha256((root/'weight_prep.cubin').read_bytes()).hexdigest()}}
 def op(entry,params,grid):return {'params':params,'impl':{'launches':[{'module':'dsv41_weight_prep','entry':entry,'block':[256,1,1],'grid':[(grid+255)//256,1,1]}]}}
 # Scalars remain public for one source cubin to cover routed/shared W1/W2.
 suffix='shared' if row_group==32 else 'routed'
 ops={f'dsv41_interleave_gate_up_{suffix}':op('dsv41_interleave_gate_up',['out buffer<u8>']+(['in buffer<fp8e4m3>']*2 if row_group==32 else ['in buffer<i8>']*2)+['i32','i32'],2*n*(k if row_group==32 else k//2)), f'dsv41_pack_expert_sf_{suffix}_{"gate_up" if gate_up else "down"}':op('dsv41_pack_expert_sf',['out buffer<i32>','in buffer<fp8e8m0>','in buffer<fp8e8m0>','i32','i32','i32','i32'],n*(2 if gate_up else 1)*(k//128))}
 return modules,ops
