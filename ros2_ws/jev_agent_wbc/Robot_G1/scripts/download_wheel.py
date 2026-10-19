#!/usr/bin/env python3
"""Download one official PyPI wheel in byte ranges, checking its PyPI SHA256."""
import concurrent.futures,hashlib,json,sys,urllib.request
from pathlib import Path
name,version=sys.argv[1:3]
root=Path(__file__).resolve().parents[1]/'.cache/wheels';root.mkdir(parents=True,exist_ok=True)
meta=json.load(urllib.request.urlopen(f'https://pypi.org/pypi/{name}/{version}/json',timeout=60))
candidates=[r for r in meta['urls'] if r['packagetype']=='bdist_wheel' and ('cp311' in r['filename'] or 'py3-none' in r['filename']) and ('manylinux' in r['filename'] and 'x86_64' in r['filename'])]
if not candidates:raise SystemExit('No compatible Linux x86_64 Python3.11 wheel')
r=candidates[0];size=r['size'];dst=root/r['filename'];parts=12
if dst.exists() and hashlib.sha256(dst.read_bytes()).hexdigest()==r['digests']['sha256']:print(dst);raise SystemExit(0)
def download(i):
    begin=size*i//parts;end=size*(i+1)//parts-1
    req=urllib.request.Request(r['url'],headers={'Range':f'bytes={begin}-{end}'})
    with urllib.request.urlopen(req,timeout=180) as response:
        if response.status!=206:raise RuntimeError('Range unsupported')
        b=response.read()
    assert len(b)==end-begin+1,(i,len(b),end-begin+1)
    return b
with concurrent.futures.ThreadPoolExecutor(max_workers=parts) as pool:blob=b''.join(pool.map(download,range(parts)))
assert len(blob)==size and hashlib.sha256(blob).hexdigest()==r['digests']['sha256']
dst.write_bytes(blob);(root/(r['filename']+'.source.json')).write_text(json.dumps({'url':r['url'],'sha256':r['digests']['sha256'],'size':size},indent=2)+'\n');print(dst)
