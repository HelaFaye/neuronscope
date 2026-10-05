#!/usr/bin/env python3
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, os, re, secrets, shutil, socket, sys, time, urllib.error, urllib.request
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

VERSION='1.0'
CHUNK=4*1024*1024
MAX_NAME=240
ID_RE=re.compile(r'^[A-Za-z0-9_-]{12,64}$')

def sha256_file(path: Path, chunk=8*1024*1024):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while True:
            b=f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def sha256_bytes(b: bytes): return hashlib.sha256(b).hexdigest()

def safe_name(name: str) -> str:
    name=Path(name).name.replace('\\','_')
    name=''.join(c if c.isprintable() and c not in '\r\n\x00' else '_' for c in name)
    if not name or name in {'.','..'}: name='model.gguf'
    return name[:MAX_NAME]

def token_ok(header, expected):
    if not expected: return True
    return header == f'Bearer {expected}'

class TransferError(RuntimeError): pass

class Receiver:
    def __init__(self, root: Path):
        self.root=root.resolve(); self.root.mkdir(parents=True, exist_ok=True)
        self.meta_dir=self.root/'.neuronscope-transfers'; self.meta_dir.mkdir(exist_ok=True)
    def paths(self, tid): return self.meta_dir/f'{tid}.json', self.meta_dir/f'{tid}.part'
    def init(self, payload):
        name=safe_name(str(payload.get('name','model.gguf')))
        size=int(payload['size']); sha=str(payload['sha256']).lower()
        if size<0 or not re.fullmatch(r'[0-9a-f]{64}',sha): raise TransferError('invalid size/sha256')
        for mp in self.meta_dir.glob('*.json'):
            try:
                old=json.loads(mp.read_text())
                if old.get('status') == 'receiving' and old.get('name') == name and int(old.get('size',-1)) == size and old.get('sha256') == sha:
                    _, pp = self.paths(old['id'])
                    old['offset'] = pp.stat().st_size if pp.exists() else 0
                    old['updated'] = time.time()
                    self.save(old)
                    return old
            except Exception:
                continue
        tid=secrets.token_urlsafe(12).replace('-','_').replace('/','_')
        meta={'version':VERSION,'id':tid,'name':name,'size':size,'sha256':sha,'created':time.time(),'updated':time.time(),'offset':0,'status':'receiving','subdir':safe_name(str(payload.get('subdir',''))) if payload.get('subdir') else ''}
        mp,pp=self.paths(tid)
        pp.touch()
        meta['offset']=pp.stat().st_size
        mp.write_text(json.dumps(meta,indent=2))
        return meta
    def get(self, tid):
        if not ID_RE.match(tid): raise TransferError('bad id')
        mp,pp=self.paths(tid)
        if not mp.exists(): raise FileNotFoundError(tid)
        return json.loads(mp.read_text())
    def save(self, meta): self.paths(meta['id'])[0].write_text(json.dumps(meta,indent=2))
    def chunk(self, tid, start, total, body, digest):
        meta=self.get(tid)
        if meta['status'] != 'receiving': raise TransferError('transfer not receiving')
        if total != meta['size']: raise TransferError('total mismatch')
        if start != int(meta['offset']): raise TransferError(f'offset mismatch; expected {meta["offset"]}, got {start}')
        if start+len(body)>total: raise TransferError('chunk exceeds total')
        if sha256_bytes(body) != digest.lower(): raise TransferError('chunk sha256 mismatch')
        mp,pp=self.paths(tid)
        with pp.open('ab') as f:
            f.write(body); f.flush(); os.fsync(f.fileno())
        meta['offset']=start+len(body); meta['updated']=time.time(); self.save(meta)
        return meta
    def complete(self, tid):
        meta=self.get(tid); mp,pp=self.paths(tid)
        if meta['offset'] != meta['size']: raise TransferError('size incomplete')
        actual=sha256_file(pp)
        if actual != meta['sha256']: raise TransferError('final sha256 mismatch')
        sub=Path(meta.get('subdir') or '')
        dest=(self.root/sub/safe_name(meta['name'])).resolve()
        if self.root not in dest.parents and dest != self.root: raise TransferError('unsafe destination')
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp=dest.with_name(dest.name+'.ns-incoming')
        os.replace(pp,temp); os.replace(temp,dest)
        meta.update(status='completed',offset=meta['size'],completed_at=time.time(),path=str(dest))
        self.save(meta)
        return meta
    def cancel(self, tid):
        meta=self.get(tid); mp,pp=self.paths(tid)
        meta['status']='cancelled'; meta['updated']=time.time(); self.save(meta)
        try: pp.unlink()
        except FileNotFoundError: pass
        return meta

class Handler(BaseHTTPRequestHandler):
    receiver: Receiver=None; token=''
    def _json(self, code, obj):
        b=json.dumps(obj).encode(); self.send_response(code); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(b))); self.end_headers(); self.wfile.write(b)
    def _auth(self):
        return token_ok(self.headers.get('Authorization',''), self.token)
    def log_message(self,*args): return
    def do_GET(self):
        if not self._auth(): return self._json(401,{'error':'unauthorized'})
        try:
            if self.path=='/health': return self._json(200,{'ok':True,'version':VERSION})
            if self.path=='/v1/transfers':
                rows=[]
                for p in self.receiver.meta_dir.glob('*.json'):
                    try: rows.append(json.loads(p.read_text()))
                    except Exception: pass
                return self._json(200,{'transfers':rows})
            m=re.fullmatch(r'/v1/transfers/([^/]+)',self.path)
            if m: return self._json(200,self.receiver.get(m.group(1)))
            m=re.fullmatch(r'/v1/transfers/([^/]+)/download',self.path)
            if m:
                meta=self.receiver.get(m.group(1));
                if meta.get('status')!='completed': return self._json(409,{'error':'not completed'})
                path=Path(meta['path']); size=path.stat().st_size
                self.send_response(200); self.send_header('Content-Length',str(size)); self.send_header('Content-Disposition',f'attachment; filename="{safe_name(meta["name"])}"'); self.end_headers()
                with path.open('rb') as f: shutil.copyfileobj(f,self.wfile,CHUNK)
                return
            self._json(404,{'error':'not found'})
        except FileNotFoundError: self._json(404,{'error':'not found'})
        except Exception as e: self._json(500,{'error':str(e)})
    def do_POST(self):
        if not self._auth(): return self._json(401,{'error':'unauthorized'})
        try:
            n=int(self.headers.get('Content-Length','0')); data=json.loads(self.rfile.read(n) or b'{}')
            if self.path=='/v1/transfers/init': return self._json(201,self.receiver.init(data))
            m=re.fullmatch(r'/v1/transfers/([^/]+)/complete',self.path)
            if m: return self._json(200,self.receiver.complete(m.group(1)))
            m=re.fullmatch(r'/v1/transfers/([^/]+)/cancel',self.path)
            if m: return self._json(200,self.receiver.cancel(m.group(1)))
            self._json(404,{'error':'not found'})
        except (TransferError,KeyError,ValueError) as e: self._json(400,{'error':str(e)})
        except Exception as e: self._json(500,{'error':str(e)})
    def do_PUT(self):
        if not self._auth(): return self._json(401,{'error':'unauthorized'})
        try:
            m=re.fullmatch(r'/v1/transfers/([^/]+)/chunk',self.path)
            if not m: return self._json(404,{'error':'not found'})
            cr=self.headers.get('Content-Range','')
            mm=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',cr)
            if not mm: return self._json(400,{'error':'Content-Range required'})
            start,end,total=map(int,mm.groups()); n=end-start+1
            if n != int(self.headers.get('Content-Length','-1')): return self._json(400,{'error':'length mismatch'})
            body=self.rfile.read(n); digest=self.headers.get('X-Chunk-SHA256','')
            return self._json(200,self.receiver.chunk(m.group(1),start,total,body,digest))
        except (TransferError,KeyError,ValueError) as e: self._json(409,{'error':str(e)})
        except Exception as e: self._json(500,{'error':str(e)})

def request(base, method, path, token='', data=None, headers=None, timeout=120, raw=False):
    h={'Authorization':f'Bearer {token}'} if token else {}
    if headers: h.update(headers)
    b=None
    if data is not None:
        if raw:
            b=data
        else:
            b=json.dumps(data).encode(); h['Content-Type']='application/json'
    req=urllib.request.Request(base.rstrip('/')+path, data=b, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw=r.read(); return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raise TransferError(f'HTTP {e.code}: {e.read().decode(errors="replace")}') from e

def send_one(base, path: Path, token='', subdir='', chunk=CHUNK, timeout=300):
    path=path.resolve(); size=path.stat().st_size; sha=sha256_file(path)
    meta=request(base,'POST','/v1/transfers/init',token,{'name':path.name,'size':size,'sha256':sha,'subdir':subdir},timeout=timeout)
    tid=meta['id']; offset=int(meta['offset'])
    with path.open('rb') as f:
        f.seek(offset)
        while offset<size:
            data=f.read(min(chunk,size-offset))
            if not data: raise TransferError('unexpected EOF')
            dg=sha256_bytes(data)
            headers={'Content-Range':f'bytes {offset}-{offset+len(data)-1}/{size}','X-Chunk-SHA256':dg}
            request(base,'PUT',f'/v1/transfers/{tid}/chunk',token,data,headers=headers,timeout=timeout,raw=True)
            offset+=len(data)
    return request(base,'POST',f'/v1/transfers/{tid}/complete',token,{},timeout=timeout)

def main(argv=None):
    p=argparse.ArgumentParser(prog='ns-transfer'); sub=p.add_subparsers(dest='cmd',required=True)
    s=sub.add_parser('serve'); s.add_argument('--host',default='0.0.0.0'); s.add_argument('--port',type=int,default=8810); s.add_argument('--root',required=True); s.add_argument('--token',default='')
    s=sub.add_parser('health'); s.add_argument('--url',required=True); s.add_argument('--token',default='')
    s=sub.add_parser('send'); s.add_argument('--url',required=True); s.add_argument('--token',default=''); s.add_argument('--file',required=True); s.add_argument('--subdir',default=''); s.add_argument('--chunk-mib',type=int,default=4)
    s=sub.add_parser('batch'); s.add_argument('--url',required=True); s.add_argument('--token',default=''); s.add_argument('--dir',required=True); s.add_argument('--glob',default='*.gguf'); s.add_argument('--workers',type=int,default=2); s.add_argument('--subdir',default='')
    s=sub.add_parser('watch'); s.add_argument('--url',required=True); s.add_argument('--token',default=''); s.add_argument('--dir',required=True); s.add_argument('--glob',default='*.gguf'); s.add_argument('--state',default='.neuronscope-transfer-watch.json'); s.add_argument('--interval',type=float,default=5); s.add_argument('--workers',type=int,default=1); s.add_argument('--subdir',default='')
    s=sub.add_parser('status'); s.add_argument('--url',required=True); s.add_argument('--token',default='')
    s=sub.add_parser('pull'); s.add_argument('--url',required=True); s.add_argument('--token',default=''); s.add_argument('--id',required=True); s.add_argument('--out',required=True)
    a=p.parse_args(argv)
    if a.cmd=='serve':
        Handler.receiver=Receiver(Path(a.root)); Handler.token=a.token
        print(f'NeuronScope Transfer Receiver listening on {a.host}:{a.port}; root={Path(a.root).resolve()}')
        ThreadingHTTPServer((a.host,a.port),Handler).serve_forever(); return 0
    if a.cmd=='health': print(json.dumps(request(a.url,'GET','/health',a.token),indent=2)); return 0
    if a.cmd=='send': print(json.dumps(send_one(a.url,Path(a.file),a.token,a.subdir,max(1,a.chunk_mib)*1024*1024),indent=2)); return 0
    if a.cmd=='batch':
        files=sorted(Path(a.dir).glob(a.glob));
        if not files: print('No matching files.'); return 0
        def f(p):
            try: return {'file':str(p),'ok':True,'result':send_one(a.url,p,a.token,a.subdir)}
            except Exception as e: return {'file':str(p),'ok':False,'error':str(e)}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,a.workers)) as ex: rows=list(ex.map(f,files))
        print(json.dumps(rows,indent=2)); return 0 if all(r['ok'] for r in rows) else 2
    if a.cmd=='watch':
        sp=Path(a.state); done={} ; stable={}
        if sp.exists():
            try:
                raw=json.loads(sp.read_text()); done=dict(raw.get('done',{}))
                if isinstance(done,list): done={p:'legacy' for p in done}
            except Exception: pass
        while True:
            files=sorted(Path(a.dir).glob(a.glob)); todo=[]
            for pth in files:
                if not pth.is_file(): continue
                key=str(pth.resolve()); st=pth.stat(); sig=(st.st_size,st.st_mtime_ns)
                previous=stable.get(key)
                stable[key]=sig
                if previous != sig: continue
                if key in done:
                    # Re-send if the content changed since the previous successful transfer.
                    if done[key] == f'{st.st_size}:{st.st_mtime_ns}':
                        continue
                todo.append(pth)
            if todo:
                def f(pth):
                    try: return str(pth.resolve()), send_one(a.url,pth,a.token,a.subdir)
                    except Exception as e: return str(pth.resolve()), {'error':str(e)}
                with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,a.workers)) as ex:
                    for key,res in ex.map(f,todo):
                        print(json.dumps({'file':key,'result':res},indent=2),flush=True)
                        if 'error' not in res:
                            st=Path(key).stat(); done[key]=f'{st.st_size}:{st.st_mtime_ns}'
                sp.write_text(json.dumps({'done':done},indent=2))
            time.sleep(max(0.5,a.interval))
    if a.cmd=='status': print(json.dumps(request(a.url,'GET','/v1/transfers',a.token),indent=2)); return 0
    if a.cmd=='pull':
        dest=Path(a.out); dest.mkdir(parents=True,exist_ok=True); meta=request(a.url,'GET',f'/v1/transfers/{a.id}',a.token); url=a.url.rstrip('/')+f'/v1/transfers/{a.id}/download'; req=urllib.request.Request(url,headers={'Authorization':f'Bearer {a.token}'} if a.token else {})
        tmp=dest/(safe_name(meta['name'])+'.ns-incoming')
        with urllib.request.urlopen(req,timeout=300) as r, tmp.open('wb') as f: shutil.copyfileobj(r,f,CHUNK)
        os.replace(tmp,dest/safe_name(meta['name'])); print(json.dumps(meta,indent=2)); return 0
    return 1

if __name__=='__main__': raise SystemExit(main())
