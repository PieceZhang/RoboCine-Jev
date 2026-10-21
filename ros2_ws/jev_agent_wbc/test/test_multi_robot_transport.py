import json
from types import SimpleNamespace
import urllib.error
import urllib.request
import pytest
from jev_agent_wbc.multi_robot_transport import OfficialPersistentTransport
from jev_agent_wbc.jev import SystemOneClient,JevProtocolError,JevTransportUnavailable,choice


class Response:
    def __init__(self,body,status=200):
        self.content=json.dumps(body).encode();self.status_code=status
        self.reason='fixture';self.headers={}
    def __enter__(self):return self
    def __exit__(self,*args):pass


def transport(monkeypatch,reply):
    calls=[]
    def post(url,**kwargs):
        calls.append((url,kwargs));return Response(reply)
    session=SimpleNamespace(post=post,close=lambda:None)
    monkeypatch.setattr('requests.Session',lambda:session)
    return OfficialPersistentTransport(),calls


def test_reuse_preserves_core_typed_validation_and_redacted_journal(monkeypatch,tmp_path):
    reply=dict(model='jev-1.14.0',answers=dict(q=dict(type='choice',choice='a',probabilities=dict(a=1.))))
    connection,calls=transport(monkeypatch,reply)
    key='unit-fixture-key-not-a-real-secret'
    keyfile=tmp_path/'key';keyfile.write_text(key)
    monkeypatch.delenv('TYPESAFE_API_KEY',raising=False)
    monkeypatch.setenv('TYPESAFE_API_KEY_FILE',str(keyfile))
    journal=tmp_path/'api.jsonl'
    client=SystemOneClient(url=connection.URL,model='jev-1.14.0',journal=journal)
    client.open_request=connection.open
    for _ in range(2):
        answer=client._ask_once(dict(stage='fixture'),dict(q=choice('fixture',dict(a='a'))))
        assert answer['q']['choice']=='a'
    assert len(calls)==2 and all(c[1]['allow_redirects'] is False for c in calls)
    assert all(c[1]['timeout']==3. and c[1]['headers']['Authorization']=='Bearer '+key for c in calls)
    assert key not in journal.read_text()
    reply['model']='wrong-model'
    with pytest.raises(JevProtocolError):
        client._ask_once({},dict(q=choice('fixture',dict(a='a'))))


@pytest.mark.parametrize('url',['http://127.0.0.1/v1/systemone','https://example.com/v1/systemone'])
def test_persistent_transport_never_redirects_credentials(monkeypatch,url):
    connection,calls=transport(monkeypatch,{})
    request=urllib.request.Request(url,data=b'{}',method='POST')
    with pytest.raises(ValueError):connection.open(request,3.)
    assert calls==[]


def test_http_retry_status_is_preserved(monkeypatch):
    connection,_=transport(monkeypatch,{})
    connection.session.post=lambda *a,**k:Response(dict(error='rate limit'),429)
    request=urllib.request.Request(connection.URL,data=b'{}',method='POST')
    with pytest.raises(urllib.error.HTTPError) as error:connection.open(request,3.)
    assert error.value.code==429 and json.load(error.value)['error']=='rate limit'
