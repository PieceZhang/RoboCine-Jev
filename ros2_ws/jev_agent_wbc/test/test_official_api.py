import json
import io
import urllib.error
import urllib.request

import pytest

from jev_agent_wbc.jev import JevError, SystemOneClient, _NoAuthRedirect


URL = 'https://api.typesafe.ai/v1/systemone'


def test_official_auth_not_journaled_and_key_file(tmp_path, monkeypatch):
    key = 'test-official-secret'
    (tmp_path/'key').write_text(key+'\n')
    monkeypatch.delenv('TYPESAFE_API_KEY', raising=False)
    monkeypatch.setenv('TYPESAFE_API_KEY_FILE', str(tmp_path/'key'))
    journal = tmp_path/'api.jsonl'

    class Opener:
        def open(self, request, timeout):
            assert request.get_header('Authorization') == 'Bearer '+key
            assert json.loads(request.data)['model'] == 'jev-1.14.0'
            raise RuntimeError('server echoed '+key)

    monkeypatch.setattr(urllib.request, 'build_opener', lambda *a: Opener())
    client = SystemOneClient(URL, 'jev-1.14.0', journal=journal)
    with pytest.raises(JevError) as caught:
        client.ask({}, {})
    assert key not in str(caught.value)
    assert key not in journal.read_text()
    assert '[REDACTED]' in journal.read_text()
    assert client.calls == 1


def test_local_adapter_never_receives_official_key(monkeypatch):
    monkeypatch.setenv('TYPESAFE_API_KEY', 'test-secret')
    assert SystemOneClient()._api_key is None
    assert SystemOneClient('https://example.com/v1/systemone')._api_key is None


def test_official_requires_valid_key(monkeypatch):
    monkeypatch.delenv('TYPESAFE_API_KEY_FILE', raising=False)
    for key in ('', 'invalid key', 'invalid\nkey'):
        monkeypatch.setenv('TYPESAFE_API_KEY', key)
        with pytest.raises(JevError, match='requires TYPESAFE'):
            SystemOneClient(URL)


def test_authenticated_redirect_refused():
    with pytest.raises(JevError, match='redirects are refused'):
        _NoAuthRedirect().redirect_request(None, None, 302, '', {}, 'https://example.com')


def test_invalid_probability_reply_preserved_without_relaxing_check(tmp_path, monkeypatch):
    reply = dict(model='test', answers={'strategy': dict(type='choice', choice='a',
                                                       probabilities={'a': .5, 'b': .4})})
    monkeypatch.setattr(urllib.request, 'urlopen', lambda *a, **kw: io.BytesIO(json.dumps(reply).encode()))
    journal = tmp_path/'api.jsonl'
    client = SystemOneClient(journal=journal)
    with pytest.raises(JevError, match='do not sum to one'):
        client.ask({}, {'strategy': dict(type='choice', criteria={'a':None, 'b':None})})
    record = json.loads(journal.read_text())
    assert record['response'] == reply
    assert 'error' in record
