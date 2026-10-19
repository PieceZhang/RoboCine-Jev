#!/usr/bin/env python3
"""Mirror NVIDIA's versioned G1 folder and AGILE's exact velocity policy."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib,json,urllib.request,xml.etree.ElementTree as E
ROOT=Path(__file__).resolve().parents[1]
BASE='https://omniverse-content-production.s3-us-west-2.amazonaws.com/'
PREFIX='Assets/Isaac/5.1/Isaac/Robots/Unitree/G1/'
EXPECTED={}
POLICY='https://media.githubusercontent.com/media/nvidia-isaac/WBC-AGILE/ad4ff6ee6b38da207592b3e71dcee4c118049f6d/agile/data/policy/velocity_g1/unitree_g1_velocity_history.pt'
def get(item):
    url,dst=item; dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists():
        with urllib.request.urlopen(url,timeout=180) as r: data=r.read()
        dst.write_bytes(data)
    print(dst.relative_to(ROOT),dst.stat().st_size,flush=True)
    digest=hashlib.sha256(dst.read_bytes()).hexdigest()
    previous=EXPECTED.get(str(dst.relative_to(ROOT)))
    if previous and digest!=previous['sha256']: raise RuntimeError('SHA256 differs from pinned asset manifest: '+str(dst))
    return dict(url=url,path=str(dst.relative_to(ROOT)),size=dst.stat().st_size,sha256=digest)
def main():
    global EXPECTED
    manifest=ROOT/'config/assets.manifest.json'
    if manifest.exists(): EXPECTED={r['path']:r for r in json.loads(manifest.read_text())}
    listing=urllib.request.urlopen(BASE+'?list-type=2&prefix='+PREFIX,timeout=60).read()
    (ROOT/'assets/s3_listing.xml').write_bytes(listing)
    ns={'s':'http://s3.amazonaws.com/doc/2006-03-01/'}
    keys=[c.find('s:Key',ns).text for c in E.fromstring(listing).findall('s:Contents',ns)]
    files=[(BASE+k,ROOT/'assets/G1'/k.removeprefix(PREFIX)) for k in keys if '/.thumbs/' not in k]
    files.append((POLICY,ROOT/'policies/unitree_g1_velocity_history.pt'))
    for folder in ('velocity_g1','velocity_height_g1'):
        for source in (ROOT/'upstream/WBC-AGILE/agile/data/policy'/folder).iterdir():
            if source.suffix in ('.pt','.onnx') and source.name!='unitree_g1_velocity_history.pt':
                url='https://media.githubusercontent.com/media/nvidia-isaac/WBC-AGILE/ad4ff6ee6b38da207592b3e71dcee4c118049f6d/agile/data/policy/'+folder+'/'+source.name
                files.append((url,ROOT/'policies'/source.name))
    with ThreadPoolExecutor(max_workers=4) as pool: records=list(pool.map(get,files))
    (ROOT/'config/assets.manifest.json').write_text(json.dumps(records,indent=2)+'\n')
if __name__=='__main__':main()
