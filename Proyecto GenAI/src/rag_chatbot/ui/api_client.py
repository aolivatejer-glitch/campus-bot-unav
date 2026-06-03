import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ApiClientError(RuntimeError):
    pass


class ApiConnectionError(ApiClientError):
    pass


class RagApiClient:
    def __init__(self, base_url: str, *, timeout: float = 20.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def index_info(self) -> dict[str, Any]:
        return self._request("GET", "/index-info")

    def query(
        self,
        *,
        question: str,
        mode: str,
        top_k: int,
        min_score: float,
        show_chunks: bool,
    ) -> dict[str, Any]:
        payload = {
            "question": question,
            "mode": mode,
            "top_k": top_k,
            "min_score": min_score,
            "show_chunks": show_chunks,
        }
        return self._request("POST", "/query", payload)

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = None
        headers = {"Accept": "application/json"}

        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            raise ApiClientError(_read_http_error(exc)) from exc
        except URLError as exc:
            raise ApiConnectionError(
                f"No se pudo conectar con la API local en {self.base_url}: {exc.reason}"
            ) from exc

        if not body:
            return {}

        return json.loads(body)


def _read_http_error(exc: HTTPError) -> str:
    body = exc.read().decode("utf-8", errors="replace")
    if not body:
        return f"API error {exc.code}"

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return body

    detail = payload.get("detail")
    if isinstance(detail, str):
        return detail

    return json.dumps(payload, ensure_ascii=False)
