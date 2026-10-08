import ctypes as C, json, pathlib, sys, torch
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from glm53 import ops_moe_v2
torch.cuda.set_device(0); torch.manual_seed(124)
cuda = C.CDLL("libcuda.so.1")
def check(e):
    if e: raise RuntimeError(e)
cache = {}
def fn_for(name, cubin, smem):
    if name not in cache:
        mod = C.c_void_p(); check(cuda.cuModuleLoad(C.byref(mod), str(cubin).encode()))
        f = C.c_void_p(); check(cuda.cuModuleGetFunction(C.byref(f), mod, name.encode()))
        if smem > 48000: check(cuda.cuFuncSetAttribute(f, 8, smem))
        cache[name] = f
    return cache[name]
op = ops_moe_v2._op(32, "tokens", resolve=True)
dt = {"f32": torch.float32, "i32": torch.int32, "bf16": torch.bfloat16, "fp8e4m3": torch.float8_e4m3fn}
sc = {n: torch.zeros(d["shape"], dtype=dt[d["dtype"]], device="cuda") for n, d in op["impl"]["scratch"].items()}
p = [torch.randn((32,4096),device="cuda",dtype=torch.bfloat16),
     (torch.randn((288,4096),device="cuda",dtype=torch.bfloat16)*.03),
     (torch.randn((288,),device="cuda")*.1),
     torch.randn((289,512,4096),device="cuda",dtype=torch.bfloat16).to(torch.float8_e4m3fn),
     torch.rand((289,4,32),device="cuda")*.03+.01,
     torch.randn((289,4096,256),device="cuda",dtype=torch.bfloat16).to(torch.float8_e4m3fn),
     torch.rand((289,32,2),device="cuda")*.06+.01,
     torch.ones((32,),device="cuda",dtype=torch.int32),
     torch.zeros((32,4096),device="cuda",dtype=torch.bfloat16)]
def ev(e,b):
    if isinstance(e,int): return e
    if isinstance(e,str): return b
    if "mul" in e: return ev(e["mul"][0],b)*ev(e["mul"][1],b)
    raise ValueError(e)
def launch(L,b):
    vals=[]
    for ty,a in zip(L["params"],L["args"]):
        if "param" in a: vals.append(C.c_uint64(p[a["param"]].data_ptr()))
        elif "scratch" in a: vals.append(C.c_uint64(sc[a["scratch"]].data_ptr()))
        elif "var" in a: vals.append(C.c_int32(b))
        elif "i32" in a: vals.append(C.c_int32(a["i32"]))
        elif "i64" in a: vals.append(C.c_uint64(a["i64"]))
        elif "i32" == ty: vals.append(C.c_int32(0))
        else: raise ValueError(a)
    argv=(C.c_void_p*len(vals))(*[C.addressof(v) for v in vals])
    grid=[ev(v,b) for v in L["grid"]]
    f=fn_for(L["entry"], L["cubin"], L.get("shared_mem",0))
    check(cuda.cuLaunchKernel(f,*grid,*L["block"],L.get("shared_mem",0),
          C.c_void_p(torch.cuda.current_stream().cuda_stream),argv,None))
for M in (3,6,12,24,32):
    for L in op["impl"]["launches"]: launch(L,M)   # fill route state
    torch.cuda.synchronize()
    res={}
    for L in op["impl"]["launches"]:
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): launch(L,M)
        for _ in range(5): g.replay()
        torch.cuda.synchronize()
        s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(50): g.replay()
        e.record(); e.synchronize()
        res[L["entry"]]=round(s.elapsed_time(e)*1000/50,2)
    res["npost_tiles"]=int(sc["npost"][0])//16
    print(M, json.dumps(res), flush=True)
