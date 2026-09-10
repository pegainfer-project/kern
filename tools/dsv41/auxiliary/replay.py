"""Replay dumped ABI cases with kern program_io, requiring byte-exact output."""
import argparse
import json
from pathlib import Path
import subprocess

p=argparse.ArgumentParser()
p.add_argument("runner",type=Path)
p.add_argument("cubins",type=Path)
p.add_argument("cases",type=Path)
a=p.parse_args()
for case in sorted(a.cases.iterdir()):
    io=json.loads((case/"io.json").read_text())
    cmd=[str(a.runner),"--manifest",str(case/"manifest.json"),"--cubins",str(a.cubins)]
    for name,path in io["inputs"].items():cmd += ["--in",f"{name}={case/path}"]
    for name in io["outputs"]:cmd += ["--out",f"{name}={case/(name+'.kern.bin')}"]
    subprocess.run(cmd,check=True)
    for name,path in io["outputs"].items():
        assert (case/path).read_bytes()==(case/(name+'.kern.bin')).read_bytes(),(case,name)
    print(f"{case.name}: byte-exact kern replay",flush=True)
