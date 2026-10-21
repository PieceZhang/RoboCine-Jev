"""Official HTTPS connection reuse below the existing typed Jev client.

Request serialization, key loading, retries, reply validation and redacted
journaling remain in SystemOneClient. This adapter only replaces socket opening.
"""
from io import BytesIO
import urllib.error


class OfficialPersistentTransport:
    URL='https://api.typesafe.ai/v1/systemone'

    def __init__(self):
        import requests
        self.session=requests.Session()

    def open(self, request, timeout):
        import requests
        if request.full_url!=self.URL or request.get_method()!='POST':
            raise ValueError('persistent authenticated transport requires the exact official endpoint')
        try:
            response=self.session.post(self.URL,data=request.data,
                headers=dict(request.header_items()),timeout=timeout,allow_redirects=False)
            body=response.content  # Consume the complete response before reusing its connection.
        except requests.exceptions.Timeout:
            raise TimeoutError('official HTTPS response timeout') from None
        except requests.exceptions.RequestException as error:
            raise urllib.error.URLError(str(error)) from None
        with response:
            if not 200<=response.status_code<300:
                raise urllib.error.HTTPError(self.URL,response.status_code,response.reason,
                    response.headers,BytesIO(body))
            return BytesIO(body)

    def close(self):
        self.session.close()
