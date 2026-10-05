#!/usr/bin/env python3
"""Optional MCP control plane for NeuronScope transfers."""
from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path

try:
    from mcp.server.fastmcp import FastMCP
except Exception as e:
    FastMCP=None; IMPORT_ERROR=e

ROOT=Path(__file__).resolve().parent
CLI=ROOT/'ns_transfer.py'
STATE=Path(os.environ.get('NS_TRANSFER_MCP_STATE', str(Path.home()/'.neuronscope-transfer-mcp.json')))

def run(cmd, timeout=30):
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout)
    return {'returncode':p.returncode,'stdout':p.stdout[-12000:],'stderr':p.stderr[-12000:]}

def load():
    if STATE.exists():
        try:return json.loads(STATE.read_text())
        except Exception:pass
    return {'targets':{}}
def save(d):
    STATE.parent.mkdir(parents=True,exist_ok=True); tmp=STATE.with_suffix('.tmp'); tmp.write_text(json.dumps(d,indent=2)); os.replace(tmp,STATE)

def ensure():
    if FastMCP is None: raise RuntimeError(f'mcp package unavailable: {IMPORT_ERROR}')

if FastMCP:
    mcp=FastMCP('neuronscope-transfer')
    @mcp.tool()
    def register_target(name:str,url:str,token:str='',destination_hint:str='')->dict:
        """Register a remote NeuronScope/LM Studio transfer receiver."""
        d=load(); d['targets'][name]={'url':url,'token':token,'destination_hint':destination_hint}; save(d); return {'ok':True,'name':name,'target':d['targets'][name]}
    @mcp.tool()
    def list_targets()->dict:
        """List configured transfer targets. Tokens are redacted."""
        d=load(); return {'targets':{k:{**v,'token':'***' if v.get('token') else ''} for k,v in d['targets'].items()}}
    @mcp.tool()
    def probe_target(name:str)->dict:
        """Check a target receiver."""
        t=load()['targets'][name]; return run([sys.executable,str(CLI),'health','--url',t['url'],'--token',t.get('token','')])
    @mcp.tool()
    def send_file(target:str,file_path:str,subdir:str='')->dict:
        """Start one reliable resumable file transfer."""
        t=load()['targets'][target]; return run([sys.executable,str(CLI),'send','--url',t['url'],'--token',t.get('token',''),'--file',file_path,'--subdir',subdir],timeout=86400)
    @mcp.tool()
    def send_batch(target:str,directory:str,glob_pattern:str='*.gguf',workers:int=2,subdir:str='')->dict:
        """Transfer a batch of intermediate GGUFs to an LM Studio host."""
        t=load()['targets'][target]; return run([sys.executable,str(CLI),'batch','--url',t['url'],'--token',t.get('token',''),'--dir',directory,'--glob',glob_pattern,'--workers',str(max(1,min(workers,8))),'--subdir',subdir],timeout=86400)
    @mcp.tool()
    def start_watch(target:str,directory:str,glob_pattern:str='*.gguf',workers:int=1,subdir:str='')->dict:
        """Start a background watcher that automatically sends new GGUFs."""
        t=load()['targets'][target]; logdir=Path.home()/'.local/state/neuronscope-transfer'; logdir.mkdir(parents=True,exist_ok=True); log=logdir/f'{target}.log'
        cmd=[sys.executable,str(CLI),'watch','--url',t['url'],'--token',t.get('token',''),'--dir',directory,'--glob',glob_pattern,'--workers',str(max(1,min(workers,8))),'--subdir',subdir]
        with log.open('a') as lf: p=subprocess.Popen(cmd,stdout=lf,stderr=subprocess.STDOUT,start_new_session=True)
        return {'ok':True,'pid':p.pid,'log':str(log),'command':cmd}
    @mcp.tool()
    def transfer_status(target:str)->dict:
        """List receiver-side transfers."""
        t=load()['targets'][target]; return run([sys.executable,str(CLI),'status','--url',t['url'],'--token',t.get('token','')])
    @mcp.tool()
    def pull_completed(target:str,transfer_id:str,out_directory:str)->dict:
        """Pull a completed remote transfer back to the controller."""
        t=load()['targets'][target]; return run([sys.executable,str(CLI),'pull','--url',t['url'],'--token',t.get('token',''),'--id',transfer_id,'--out',out_directory],timeout=86400)
else:
    mcp=None

def main():
    ensure(); mcp.run()

if __name__=='__main__': main()
