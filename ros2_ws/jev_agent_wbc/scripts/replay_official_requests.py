#!/usr/bin/env python3
"""Pair historical adapter requests with official Jev on identical state/questions.

Replay choices are diagnostic only; no robot execution or new success claims.
"""
import argparse
import json
from pathlib import Path
import time
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--key-file', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    key = a.key_file.read_text().strip()
    from jev_agent_wbc.jev import _NoAuthRedirect
    opener = urllib.request.build_opener(_NoAuthRedirect())
    with a.output.open('x') as stream:
        for source in sorted(a.baseline.glob('*_jev/jev_api.jsonl')):
            for index, line in enumerate(source.read_text().splitlines()):
                previous = json.loads(line)
                request = dict(previous['request'], model='jev-1.14.0')
                record = dict(case=source.parent.name, index=index,
                              source=str(source.resolve()), request=request,
                              baseline_response=previous.get('response'),
                              baseline_error=previous.get('error'),
                              baseline_wall_s=previous['wall_s'])
                started = time.monotonic()
                try:
                    req = urllib.request.Request('https://api.typesafe.ai/v1/systemone',
                        data=json.dumps(request, allow_nan=False).encode(),
                        headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
                    with opener.open(req, timeout=3.) as response:
                        record['response'] = json.load(response)
                except Exception as e:
                    record['error'] = str(e).replace(key, '[REDACTED]')
                record['wall_s'] = time.monotonic()-started
                stream.write(json.dumps(record, ensure_ascii=False)+'\n'); stream.flush()
            print(source.parent.name, 'replayed', index+1, flush=True)


if __name__ == '__main__':
    main()
