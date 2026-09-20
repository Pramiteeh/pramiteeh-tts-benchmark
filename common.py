"""Shared dataset validation and Indic-safe ASR normalization."""
import hashlib
import json
import os
from urllib.parse import urlsplit
from pathlib import Path
import re
import unicodedata


def normalize(text):
    text = unicodedata.normalize("NFC", text).lower()
    text = "".join(c if unicodedata.category(c)[0] in "LMN" or c == "_"
                   else "" if c in "\u200c\u200d" else " " for c in text)
    return " ".join(text.split())


def read_items(path):
    items = json.loads(Path(path).read_text(encoding="utf-8"))["items"]
    if not isinstance(items, list) or not items:
        raise ValueError("Dataset must contain a nonempty items list")
    seen = set()
    for item in items:
        if not isinstance(item.get("id"), str) or not re.fullmatch(r"[A-Za-z0-9_-]+", item["id"]):
            raise ValueError("Each item needs a safe filename id")
        if item["id"] in seen:
            raise ValueError(f"Duplicate sample id: {item['id']}")
        seen.add(item["id"])
        for key in ("lang", "text"):
            if not isinstance(item.get(key), str) or not item[key].strip() or "\0" in item[key]:
                raise ValueError(f"Invalid {key} for {item['id']}")
        ref = item.get("spoken_reference", item["text"])
        if not isinstance(ref, str) or not normalize(ref):
            raise ValueError(f"Invalid spoken reference for {item['id']}")
    return items


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_tts_url(url):
    endpoint = urlsplit(url)
    if (endpoint.scheme not in ("http", "https") or not endpoint.hostname
            or endpoint.username is not None or endpoint.password is not None
            or endpoint.query or endpoint.fragment):
        raise ValueError("Use a plain HTTP(S) TTS URL without credentials, query or fragment")
    return endpoint


def tts_headers(url, env=None):
    """Private inference key; never forwarded to Sarvam or stored in results."""
    endpoint = validate_tts_url(url)
    env = os.environ if env is None else env
    key = env.get("TTS_API_KEY", "").strip()
    filename = env.get("TTS_API_KEY_FILE", "")
    if key and filename:
        raise ValueError("Set only one of TTS_API_KEY or TTS_API_KEY_FILE")
    if filename:
        key = Path(filename).read_text(encoding="utf-8").strip()
        if not key:
            raise ValueError("TTS API key file is empty")
    if not key:
        return {}  # An authenticated backend will return 401.
    if not re.fullmatch(r"[A-Za-z0-9._~+/-]{32,512}={0,2}", key):
        raise ValueError("Invalid TTS API key format")
    if endpoint.scheme != "https" and endpoint.hostname not in ("127.0.0.1", "::1", "localhost"):
        raise ValueError("Use HTTPS for a remote TTS API key")
    return {"Authorization": "Bearer " + key}
