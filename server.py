import json
import os
import re
import socketserver
import urllib.error
import urllib.parse
import urllib.request
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler
from pathlib import Path

PORT = 5000
ROOT = Path(__file__).resolve().parent

VALID_TAGS = frozenset({"movie", "anime", "game", "show", "song"})
PROVIDER_ENV = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "grok": "GROK_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
}
GEMINI_MODELS = (
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-flash-latest",
    "gemini-1.5-flash",
)


def load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv(ROOT / ".env")

DEFAULT_PROVIDER = os.environ.get("AI_PROVIDER", "gemini").strip().lower()
if DEFAULT_PROVIDER not in PROVIDER_ENV:
    DEFAULT_PROVIDER = "gemini"


def provider_keys(name: str) -> list[str]:
    env_name = PROVIDER_ENV.get(name)
    if not env_name:
        return []
    raw = os.environ.get(env_name, "")
    return [k.strip() for k in re.split(r"[,;]+", raw) if k.strip()]


def configured_providers() -> list[str]:
    return [p for p in PROVIDER_ENV if provider_keys(p)]


def total_env_key_count() -> int:
    return sum(len(provider_keys(p)) for p in PROVIDER_ENV)


def classify_prompt(q: str) -> str:
    return f"""You are a STRICT content classifier. Reply ONLY with valid JSON, no markdown, no explanation, no extra text.
Format: {{"tags":["game"]}}
Available tags: movie, anime, game, show, song
Rules:
- VIDEO GAME (any title, sequel, franchise, mobile/PC/console game) = ["game"] ONLY. Never add "movie".
- ANIME / MANGA / Japanese animation = ["anime"] ONLY
- MOVIE / FILM (theatrical release, not a game) = ["movie"] ONLY
- TV show / web series / season / episode = ["show"] ONLY
- SONG / MUSIC / album / artist / lyrics / track / single = ["song"] ONLY
- Never return all 5 tags. Default to a single most-likely tag.
Classify this: "{q}" """


def extract_tags(text: str) -> list[str] | None:
    if not text:
        return None
    t = re.sub(r"```[a-z]*|```", "", str(text), flags=re.I).strip()
    try:
        parsed = json.loads(t)
        if isinstance(parsed.get("tags"), list):
            return parsed["tags"]
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    mobj = re.search(r"\{[\s\S]*?\}", t)
    if mobj:
        try:
            parsed = json.loads(mobj.group(0))
            if isinstance(parsed.get("tags"), list):
                return parsed["tags"]
        except json.JSONDecodeError:
            pass
    amobj = re.search(r"\[[\s\S]*?\]", t)
    if amobj:
        try:
            parsed = json.loads(amobj.group(0))
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
    found = [v for v in VALID_TAGS if re.search(rf"\b{v}\b", t, re.I)]
    return found or None


def clean_tags(tags) -> list[str] | None:
    if not isinstance(tags, list):
        return None
    cleaned = list(
        dict.fromkeys(
            str(t).lower().strip()
            for t in tags
            if str(t).lower().strip() in VALID_TAGS
        )
    )
    if not cleaned or len(cleaned) == 5:
        return None
    return cleaned


def is_key_failure_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return bool(
        re.search(
            r"quota|rate.?limit|billing|invalid.*api|api[_ ]?key|exhausted|"
            r"permission|denied|unauthorized|exceeded|resource_exhausted|credit|401|403|429|402",
            msg,
        )
    )


def http_json(method: str, url: str, headers: dict, body: dict | None = None) -> dict:
    data = None
    hdrs = dict(headers)
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        hdrs.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        try:
            err_json = json.loads(err_body)
        except json.JSONDecodeError:
            err_json = {"error": {"message": err_body or str(e)}}
        err_obj = err_json.get("error", err_json)
        if isinstance(err_obj, dict):
            msg = err_obj.get("message") or str(err_obj)
        else:
            msg = str(err_obj) if err_obj else f"HTTP {e.code}"
        raise RuntimeError(msg) from e


def classify_openai_compat(
    q: str, api_key: str, base_url: str, model: str, json_mode: bool = True
) -> list[str] | None:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": classify_prompt(q)}],
        "max_tokens": 60,
        "temperature": 0,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    data = http_json(
        "POST",
        f"{base_url}/chat/completions",
        {"Authorization": f"Bearer {api_key}"},
        body,
    )
    if isinstance(data, dict) and data.get("error"):
        err = data["error"]
        msg = err.get("message") if isinstance(err, dict) else str(err)
        raise RuntimeError(msg)
    text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    return clean_tags(extract_tags(text))


def classify_gemini(q: str, api_key: str) -> list[str] | None:
    body = {
        "contents": [{"parts": [{"text": classify_prompt(q)}]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 60,
            "responseMimeType": "application/json",
        },
    }
    last_err = None
    for model in GEMINI_MODELS:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
            f":generateContent?key={urllib.parse.quote(api_key)}"
        )
        try:
            data = http_json("POST", url, {}, body)
            if data.get("error"):
                msg = data["error"].get("message", "Gemini error")
                if re.search(r"not found|not supported", msg, re.I):
                    last_err = RuntimeError(msg)
                    continue
                raise RuntimeError(msg)
            text = (
                (data.get("candidates") or [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )
            return clean_tags(extract_tags(text))
        except RuntimeError as e:
            last_err = e
            if not re.search(r"not found|not supported|404", str(e), re.I):
                raise
    if last_err:
        raise last_err
    return None


def classify_claude(q: str, api_key: str) -> list[str] | None:
    data = http_json(
        "POST",
        "https://api.anthropic.com/v1/messages",
        {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 60,
            "messages": [{"role": "user", "content": classify_prompt(q)}],
        },
    )
    text = (data.get("content") or [{}])[0].get("text", "")
    return clean_tags(extract_tags(text))


def classify_query_with_key(q: str, provider: str, api_key: str) -> list[str] | None:
    if provider == "gemini":
        return classify_gemini(q, api_key)
    if provider == "openai":
        return classify_openai_compat(
            q, api_key, "https://api.openai.com/v1", "gpt-4o-mini"
        )
    if provider == "groq":
        return classify_openai_compat(
            q, api_key, "https://api.groq.com/openai/v1", "llama-3.1-8b-instant"
        )
    if provider == "grok":
        return classify_openai_compat(
            q, api_key, "https://api.x.ai/v1", "grok-2-1212", json_mode=False
        )
    if provider == "claude":
        return classify_claude(q, api_key)
    raise ValueError(f"Unknown provider: {provider}")


def classify_with_env_fallback(q: str, preferred_provider: str) -> dict:
    """Try every .env key: preferred provider first, then others."""
    errors: list[str] = []
    order = [preferred_provider] + [p for p in PROVIDER_ENV if p != preferred_provider]
    for prov in order:
        for idx, key in enumerate(provider_keys(prov)):
            try:
                tags = classify_query_with_key(q, prov, key)
                if tags:
                    return {
                        "tags": tags,
                        "provider": prov,
                        "source": "env",
                        "keyIndex": idx,
                    }
            except Exception as e:
                errors.append(f"{prov}[{idx}]: {e}")
                continue
    raise RuntimeError(errors[-1] if errors else "No env API keys configured")


class NoCacheHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_GET(self):
        if self.path == "/api/ai/config":
            self._send_json(
                HTTPStatus.OK,
                {
                    "configured": total_env_key_count() > 0,
                    "provider": DEFAULT_PROVIDER,
                    "providers": configured_providers(),
                    "keyCount": total_env_key_count(),
                },
            )
            return
        super().do_GET()

    def do_POST(self):
        if self.path != "/api/ai/classify":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        try:
            body = self._read_json()
        except ValueError as e:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(e)})
            return

        q = (body.get("query") or body.get("q") or "").strip()
        if not q:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "query is required"})
            return

        provider = (body.get("provider") or DEFAULT_PROVIDER).strip().lower()
        if provider not in PROVIDER_ENV:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"invalid provider: {provider}"})
            return

        use_fallback = body.get("fallback", True)
        if not use_fallback:
            keys = provider_keys(provider)
            if not keys:
                self._send_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {"error": f"No {PROVIDER_ENV[provider]} in .env"},
                )
                return
            try:
                tags = classify_query_with_key(q, provider, keys[0])
                self._send_json(
                    HTTPStatus.OK,
                    {"tags": tags, "provider": provider, "source": "env"},
                )
            except Exception as e:
                self._send_json(
                    HTTPStatus.BAD_GATEWAY,
                    {"error": str(e), "provider": provider, "keyFailed": True},
                )
            return

        if total_env_key_count() == 0:
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": "No API keys in .env"},
            )
            return

        try:
            result = classify_with_env_fallback(q, provider)
            self._send_json(HTTPStatus.OK, result)
        except Exception as e:
            self._send_json(
                HTTPStatus.BAD_GATEWAY,
                {
                    "error": str(e),
                    "keyFailed": is_key_failure_error(e),
                    "allFailed": True,
                },
            )

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            raise ValueError("empty body")
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _send_json(self, status: HTTPStatus, payload: dict):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    count = total_env_key_count()
    with socketserver.TCPServer(("0.0.0.0", PORT), NoCacheHandler) as httpd:
        httpd.allow_reuse_address = True
        print(f"Serving on http://0.0.0.0:{PORT}")
        if count:
            print(
                f"AI backup keys from .env: {count} total "
                f"({', '.join(configured_providers())}) — default: {DEFAULT_PROVIDER}"
            )
        else:
            print("No AI keys in .env — copy .env.example to .env and add your keys")
        httpd.serve_forever()
