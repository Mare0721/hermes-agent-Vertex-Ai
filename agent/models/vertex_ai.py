"""Vertex AI Studio API-key adapter for Hermes.

This module exposes a chat.completions-like client so the rest of Hermes can
keep using its OpenAI-oriented tool loop while routing requests to Vertex
``generateContent`` / ``streamGenerateContent``.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import queue
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional, Tuple
from urllib.parse import unquote, urlparse

import httpx


logger = logging.getLogger(__name__)


_DEFAULT_REGION = "global"
_DEFAULT_TIMEOUT_SECONDS = 300.0
_MAX_INLINE_IMAGE_BYTES = 20 * 1024 * 1024


def _read_auxiliary_vision_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        vision_cfg = cfg.get("auxiliary", {}).get("vision", {})
        if isinstance(vision_cfg, dict):
            return vision_cfg
    except Exception:
        pass
    return {}


def _resolve_remote_image_timeout_seconds() -> float:
    env_val = os.getenv("HERMES_VERTEX_REMOTE_IMAGE_TIMEOUT_SECONDS", "").strip()
    if env_val:
        try:
            parsed = float(env_val)
            if parsed > 0:
                return parsed
        except ValueError:
            pass

    cfg_val = _read_auxiliary_vision_config().get("remote_image_timeout")
    if cfg_val is not None:
        try:
            parsed = float(cfg_val)
            if parsed > 0:
                return parsed
        except (TypeError, ValueError):
            pass

    return 20.0


def _resolve_remote_image_cache_ttl_seconds() -> float:
    env_val = os.getenv("HERMES_VERTEX_REMOTE_IMAGE_CACHE_TTL_SECONDS", "").strip()
    if env_val:
        try:
            parsed = float(env_val)
            if parsed > 0:
                return parsed
        except ValueError:
            pass

    cfg_val = _read_auxiliary_vision_config().get("remote_image_cache_ttl")
    if cfg_val is not None:
        try:
            parsed = float(cfg_val)
            if parsed > 0:
                return parsed
        except (TypeError, ValueError):
            pass

    return 180.0


def _resolve_remote_image_cache_max_entries() -> int:
    env_val = os.getenv("HERMES_VERTEX_REMOTE_IMAGE_CACHE_MAX_ENTRIES", "").strip()
    if env_val:
        try:
            parsed = int(float(env_val))
            if parsed > 0:
                return parsed
        except ValueError:
            pass

    cfg_val = _read_auxiliary_vision_config().get("remote_image_cache_max_entries")
    if cfg_val is not None:
        try:
            parsed = int(float(cfg_val))
            if parsed > 0:
                return parsed
        except (TypeError, ValueError):
            pass

    return 96


_REMOTE_IMAGE_TIMEOUT_SECONDS = _resolve_remote_image_timeout_seconds()
_REMOTE_IMAGE_CACHE_TTL_SECONDS = _resolve_remote_image_cache_ttl_seconds()
_REMOTE_IMAGE_CACHE_MAX_ENTRIES = _resolve_remote_image_cache_max_entries()
_REMOTE_IMAGE_CACHE_MAX_TOTAL_BYTES = 200 * 1024 * 1024
_REMOTE_IMAGE_CACHE: Dict[str, Tuple[float, Dict[str, str], str, int]] = {}
_REMOTE_IMAGE_CACHE_LOCK = threading.Lock()
_REMOTE_IMAGE_RUNTIME_CONFIG_LOGGED = False


def _log_remote_image_runtime_config_once() -> None:
    global _REMOTE_IMAGE_RUNTIME_CONFIG_LOGGED
    if _REMOTE_IMAGE_RUNTIME_CONFIG_LOGGED:
        return

    _REMOTE_IMAGE_RUNTIME_CONFIG_LOGGED = True
    logger.info(
        "Vertex remote image config: timeout=%.2fs, cache_ttl=%.2fs, cache_max_entries=%d, cache_max_total_bytes=%d",
        _REMOTE_IMAGE_TIMEOUT_SECONDS,
        _REMOTE_IMAGE_CACHE_TTL_SECONDS,
        _REMOTE_IMAGE_CACHE_MAX_ENTRIES,
        _REMOTE_IMAGE_CACHE_MAX_TOTAL_BYTES,
    )


def _normalize_region(region: Optional[str]) -> str:
    value = (region or "").strip()
    return value or _DEFAULT_REGION


def _host_for_region(region: str) -> str:
    normalized = _normalize_region(region).lower()
    if normalized == "global":
        return "https://aiplatform.googleapis.com"
    return f"https://{normalized}-aiplatform.googleapis.com"


def build_vertex_models_base_url(project_id: str, region: str) -> str:
    """Build the models collection base URL for Vertex Generative AI REST APIs."""
    project = (project_id or "").strip()
    if not project:
        return ""
    location = _normalize_region(region)
    host = _host_for_region(location)
    return (
        f"{host}/v1/projects/{project}/locations/{location}"
        "/publishers/google/models"
    )


def _parse_project_and_region_from_base_url(base_url: str) -> Tuple[str, str]:
    text = str(base_url or "").strip().rstrip("/")
    if "/projects/" not in text or "/locations/" not in text:
        return "", ""

    project_id = ""
    region = ""
    try:
        project_id = text.split("/projects/", 1)[1].split("/", 1)[0]
        region = text.split("/locations/", 1)[1].split("/", 1)[0]
    except Exception:
        return "", ""
    return project_id.strip(), region.strip()


def _coerce_timeout_seconds(timeout: Any, default: float = _DEFAULT_TIMEOUT_SECONDS) -> float:
    if timeout is None:
        return default
    if isinstance(timeout, (int, float)):
        return float(timeout)
    if isinstance(timeout, httpx.Timeout):
        read_timeout = timeout.read
        if isinstance(read_timeout, (int, float)):
            return float(read_timeout)
    return default


def _safe_json_loads(value: Any) -> Any:
    if isinstance(value, (dict, list, int, float, bool)) or value is None:
        return value
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        return {"text": text}


def _json_string(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _extract_text_from_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "").strip().lower()
            if item_type in ("text", "input_text"):
                text_value = item.get("text")
                if isinstance(text_value, str) and text_value:
                    parts.append(text_value)
        return "\n".join(p for p in parts if p)
    if isinstance(content, dict):
        text_value = content.get("text")
        if isinstance(text_value, str):
            return text_value
    return str(content)


def _data_url_to_inline_data(image_url: Any) -> Optional[Dict[str, str]]:
    """Convert data:image/*;base64 URLs into Vertex inlineData payloads."""
    text = str(image_url or "").strip()
    if not text.lower().startswith("data:"):
        return None

    header, sep, payload = text.partition(",")
    if not sep or "base64" not in header.lower():
        return None

    mime_type = header[len("data:"):].split(";", 1)[0].strip() or "image/jpeg"
    if not mime_type.startswith("image/"):
        return None

    encoded = payload.strip()
    if not encoded:
        return None

    return {"mimeType": mime_type, "data": encoded}


def _mime_type_from_suffix(suffix: str) -> Optional[str]:
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
        ".svg": "image/svg+xml",
    }.get(suffix)


def _mime_type_from_path(image_path: Path) -> str:
    return _mime_type_from_suffix(image_path.suffix.lower()) or "image/jpeg"


def _local_image_to_inline_data(image_url: Any) -> Optional[Dict[str, str]]:
    """Convert local image paths/file:// URIs into inlineData payloads."""
    raw = str(image_url or "").strip()
    if not raw or raw.lower().startswith(("http://", "https://", "data:")):
        return None

    candidate = raw
    if raw.lower().startswith("file://"):
        parsed = urlparse(raw)
        candidate = unquote(parsed.path or "")
        if os.name == "nt" and candidate.startswith("/") and len(candidate) > 2 and candidate[2] == ":":
            candidate = candidate[1:]

    path = Path(os.path.expanduser(candidate))
    if not path.is_file():
        return None
    if path.stat().st_size > _MAX_INLINE_IMAGE_BYTES:
        return None

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "mimeType": _mime_type_from_path(path),
        "data": encoded,
    }


def _detect_image_mime_from_bytes(data: bytes) -> Optional[str]:
    """Best-effort image type detection from payload bytes."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"BM"):
        return "image/bmp"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"

    head = data[:4096].decode("utf-8", errors="ignore").lower()
    if "<svg" in head:
        return "image/svg+xml"

    return None


def _cache_get_remote_image(url: str) -> Optional[Dict[str, str]]:
    now = time.time()
    with _REMOTE_IMAGE_CACHE_LOCK:
        entry = _REMOTE_IMAGE_CACHE.get(url)
        if not entry:
            return None
        expires_at, inline_data, _digest, _payload_size = entry
        if expires_at <= now:
            _REMOTE_IMAGE_CACHE.pop(url, None)
            return None
        return dict(inline_data)


def _cache_set_remote_image(url: str, inline_data: Dict[str, str], payload: bytes) -> None:
    expires_at = time.time() + _REMOTE_IMAGE_CACHE_TTL_SECONDS
    digest = hashlib.sha256(payload).hexdigest()
    payload_size = len(payload)
    with _REMOTE_IMAGE_CACHE_LOCK:
        _REMOTE_IMAGE_CACHE[url] = (expires_at, dict(inline_data), digest, payload_size)

        total_bytes = sum(item[3] for item in _REMOTE_IMAGE_CACHE.values())
        while _REMOTE_IMAGE_CACHE and (
            len(_REMOTE_IMAGE_CACHE) > _REMOTE_IMAGE_CACHE_MAX_ENTRIES
            or total_bytes > _REMOTE_IMAGE_CACHE_MAX_TOTAL_BYTES
        ):
            oldest_key, oldest_entry = min(
                _REMOTE_IMAGE_CACHE.items(),
                key=lambda item: item[1][0],
            )
            _REMOTE_IMAGE_CACHE.pop(oldest_key, None)
            total_bytes -= oldest_entry[3]


def _clear_remote_image_cache_for_tests() -> None:
    """Testing hook for deterministic cache behavior."""
    with _REMOTE_IMAGE_CACHE_LOCK:
        _REMOTE_IMAGE_CACHE.clear()


def _remote_image_url_to_inline_data(image_url: Any) -> Optional[Dict[str, str]]:
    """Convert safe remote HTTP(S) image URLs into inlineData payloads."""
    raw = str(image_url or "").strip()
    if not raw.lower().startswith(("http://", "https://")):
        return None

    _log_remote_image_runtime_config_once()

    # Respect SSRF protection and website policy controls used by other tools.
    try:
        from tools.url_safety import is_safe_url
        from tools.website_policy import check_website_access
    except Exception:
        return None

    if not is_safe_url(raw):
        return None
    blocked = check_website_access(raw)
    if blocked:
        return None

    cached = _cache_get_remote_image(raw)
    if cached:
        return cached

    def _redirect_guard(response: httpx.Response) -> None:
        if response.is_redirect and response.next_request:
            redirect_url = str(response.next_request.url)
            if not is_safe_url(redirect_url):
                raise ValueError(f"Blocked redirect URL: {redirect_url}")

    try:
        with httpx.Client(
            timeout=_REMOTE_IMAGE_TIMEOUT_SECONDS,
            follow_redirects=True,
            event_hooks={"response": [_redirect_guard]},
        ) as client:
            with client.stream(
                "GET",
                raw,
                headers={"Accept": "image/*,*/*;q=0.8"},
            ) as response:
                response.raise_for_status()
                final_url = str(response.url)
                if not is_safe_url(final_url):
                    return None
                blocked = check_website_access(final_url)
                if blocked:
                    return None

                cl = response.headers.get("content-length")
                if cl and int(cl) > _MAX_INLINE_IMAGE_BYTES:
                    return None

                header_mime = str(response.headers.get("content-type") or "").split(";", 1)[0].strip().lower()

                body = bytearray()
                for chunk in response.iter_bytes():
                    if not chunk:
                        continue
                    body.extend(chunk)
                    if len(body) > _MAX_INLINE_IMAGE_BYTES:
                        return None

            body_bytes = bytes(body)
            sniffed_mime = _detect_image_mime_from_bytes(body_bytes)
            if not sniffed_mime:
                return None

            # Use server header when it looks valid and agrees on image class,
            # otherwise trust magic-byte sniffing.
            mime_type = sniffed_mime
            if header_mime.startswith("image/") and header_mime == sniffed_mime:
                mime_type = header_mime

            encoded = base64.b64encode(body_bytes).decode("ascii")
            inline_data = {"mimeType": mime_type, "data": encoded}
            _cache_set_remote_image(raw, inline_data, body_bytes)
            if final_url and final_url != raw:
                _cache_set_remote_image(final_url, inline_data, body_bytes)
            return inline_data
    except Exception:
        return None


def _content_to_vertex_parts(content: Any) -> List[Dict[str, Any]]:
    if isinstance(content, list):
        parts: List[Dict[str, Any]] = []
        for item in content:
            if isinstance(item, str):
                text_item = item.strip()
                if text_item:
                    parts.append({"text": text_item})
                continue

            if not isinstance(item, dict):
                continue

            item_type = str(item.get("type") or "").strip().lower()
            if item_type in ("text", "input_text"):
                text_item = str(item.get("text") or "").strip()
                if text_item:
                    parts.append({"text": text_item})
                continue

            if item_type in ("image_url", "input_image"):
                image_spec = item.get("image_url")
                image_url = ""
                if isinstance(image_spec, dict):
                    image_url = str(
                        image_spec.get("url") or image_spec.get("image_url") or ""
                    ).strip()
                elif isinstance(image_spec, str):
                    image_url = image_spec.strip()
                elif isinstance(item.get("url"), str):
                    image_url = str(item.get("url") or "").strip()

                inline_data = _data_url_to_inline_data(image_url)
                if not inline_data:
                    inline_data = _local_image_to_inline_data(image_url)
                if not inline_data:
                    inline_data = _remote_image_url_to_inline_data(image_url)
                if inline_data:
                    parts.append({"inlineData": inline_data})
                elif image_url:
                    # Keep non-data URLs visible as text instead of silently dropping them.
                    parts.append({"text": f"[Image URL: {image_url}]"})
                continue

            text_item = str(item.get("text") or "").strip()
            if text_item:
                parts.append({"text": text_item})

        return parts

    text = _extract_text_from_content(content).strip()
    if not text:
        return []
    return [{"text": text}]


def _suffix_delta(current: str, previous: str) -> str:
    if not current:
        return ""
    if previous and current.startswith(previous):
        return current[len(previous):]
    return current


def _map_finish_reason(vertex_reason: Optional[str], has_tool_calls: bool = False) -> str:
    if has_tool_calls:
        return "tool_calls"

    reason = str(vertex_reason or "").strip().upper()
    if reason in ("", "STOP", "FINISH_REASON_UNSPECIFIED"):
        return "stop"
    if reason in ("MAX_TOKENS", "RECITATION"):
        return "length"
    if reason in (
        "SAFETY",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "MODEL_ARMOR",
    ):
        return "content_filter"
    return "stop"


def _usage_from_vertex(response_json: Dict[str, Any]) -> Optional[SimpleNamespace]:
    usage = response_json.get("usageMetadata")
    if not isinstance(usage, dict):
        return None

    prompt_tokens = int(usage.get("promptTokenCount") or 0)
    completion_tokens = int(
        usage.get("candidatesTokenCount")
        or usage.get("outputTokenCount")
        or usage.get("responseTokenCount")
        or 0
    )
    total_tokens = int(usage.get("totalTokenCount") or (prompt_tokens + completion_tokens))
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def _deterministic_tool_call_id(name: str, arguments: str, index: int) -> str:
    seed = f"{name}|{arguments}|{index}"
    digest = uuid.uuid5(uuid.NAMESPACE_URL, seed).hex[:16]
    return f"call_{digest}"


def _extract_text_and_tool_calls(parts: Iterable[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    text_chunks: List[str] = []
    tool_calls: List[Dict[str, str]] = []

    for idx, part in enumerate(parts):
        if not isinstance(part, dict):
            continue

        text_value = part.get("text")
        if isinstance(text_value, str) and text_value:
            text_chunks.append(text_value)

        function_call = part.get("functionCall")
        if isinstance(function_call, dict):
            name = str(function_call.get("name") or "").strip()
            args_value = function_call.get("args")
            if isinstance(args_value, str):
                arguments = args_value
            else:
                arguments = _json_string(args_value if args_value is not None else {})
            thought_signature = part.get("thought_signature") or part.get("thoughtSignature")
            if not thought_signature:
                thought_signature = function_call.get("thought_signature") or function_call.get("thoughtSignature")
            extra_content = None
            if thought_signature:
                extra_content = {"google": {"thought_signature": thought_signature}}
            tool_calls.append(
                {
                    "id": _deterministic_tool_call_id(name, arguments, idx),
                    "name": name,
                    "arguments": arguments,
                    "extra_content": extra_content,
                }
            )

    return "".join(text_chunks).strip(), tool_calls


def _normalize_tool_response_payload(raw_content: Any, tool_call_id: str) -> Dict[str, Any]:
    parsed = _safe_json_loads(raw_content)
    if isinstance(parsed, dict):
        payload: Dict[str, Any] = dict(parsed)
    else:
        payload = {"result": parsed}

    if tool_call_id and "tool_call_id" not in payload:
        payload["tool_call_id"] = tool_call_id
    return payload


def _build_vertex_messages(messages: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convert OpenAI-style messages to Vertex ``contents`` + ``systemInstruction``.

    Vertex roles are limited to ``user`` and ``model``; consecutive roles are
    merged to maintain strict alternation.
    """
    system_texts: List[str] = []
    tool_name_by_id: Dict[str, str] = {}
    thought_signature_by_call_id: Dict[str, Any] = {}
    contents: List[Dict[str, Any]] = []

    def append_parts(role: str, parts: List[Dict[str, Any]]) -> None:
        if not parts:
            return
        if contents and contents[-1].get("role") == role:
            contents[-1]["parts"].extend(parts)
            return
        contents.append({"role": role, "parts": list(parts)})

    for message in messages or []:
        if not isinstance(message, dict):
            continue

        role = str(message.get("role") or "").strip().lower()
        if role in ("system", "developer"):
            text = _extract_text_from_content(message.get("content")).strip()
            if text:
                system_texts.append(text)
            continue

        if role == "assistant":
            parts = _content_to_vertex_parts(message.get("content"))
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") or {}
                if not isinstance(fn, dict):
                    continue
                name = str(fn.get("name") or "").strip()
                arguments = fn.get("arguments")
                args_obj = _safe_json_loads(arguments)
                if not isinstance(args_obj, dict):
                    args_obj = {"_raw": args_obj}

                function_call_part: Dict[str, Any] = {"functionCall": {"name": name, "args": args_obj}}

                extra_content = call.get("extra_content")
                thought_signature = None
                if isinstance(extra_content, dict):
                    google_blob = extra_content.get("google")
                    if isinstance(google_blob, dict):
                        thought_signature = google_blob.get("thought_signature")
                    if not thought_signature:
                        thought_signature = extra_content.get("thought_signature")
                if thought_signature:
                    # Canonical Vertex JSON field for GenerateContent parts.
                    function_call_part["thoughtSignature"] = thought_signature

                parts.append(function_call_part)

                call_id = str(call.get("id") or "").strip()
                if call_id and name:
                    tool_name_by_id[call_id] = name
                    if thought_signature:
                        thought_signature_by_call_id[call_id] = thought_signature

            append_parts("model", parts)
            continue

        if role == "tool":
            tool_call_id = str(message.get("tool_call_id") or "").strip()
            tool_name = str(message.get("name") or "").strip()
            if not tool_name:
                tool_name = tool_name_by_id.get(tool_call_id, "tool")
            response_payload = _normalize_tool_response_payload(message.get("content"), tool_call_id)
            function_response_part: Dict[str, Any] = {
                "functionResponse": {"name": tool_name, "response": response_payload}
            }
            append_parts(
                "user",
                [function_response_part],
            )
            continue

        if role == "user":
            parts = _content_to_vertex_parts(message.get("content"))
            append_parts("user", parts)
            continue

    if not contents:
        contents = [{"role": "user", "parts": [{"text": ""}]}]

    system_instruction = None
    if system_texts:
        system_instruction = {"parts": [{"text": "\n\n".join(system_texts)}]}

    return system_instruction, contents


def _build_vertex_tools(tools: Any) -> List[Dict[str, Any]]:
    declarations: List[Dict[str, Any]] = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") != "function":
            continue
        fn = tool.get("function") or {}
        if not isinstance(fn, dict):
            continue

        name = str(fn.get("name") or "").strip()
        if not name:
            continue

        declaration: Dict[str, Any] = {"name": name}
        description = fn.get("description")
        if isinstance(description, str) and description.strip():
            declaration["description"] = description.strip()

        parameters = fn.get("parameters")
        if isinstance(parameters, dict) and parameters:
            declaration["parameters"] = parameters

        declarations.append(declaration)

    if len(declarations) > 128:
        declarations = declarations[:128]

    if not declarations:
        return []
    return [{"functionDeclarations": declarations}]


def _build_vertex_native_search_tool(kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    tool_choice = kwargs.get("tool_choice")
    if isinstance(tool_choice, str) and tool_choice.strip().lower() == "none":
        return None

    containers: List[Dict[str, Any]] = [kwargs]
    extra_body = kwargs.get("extra_body")
    if isinstance(extra_body, dict):
        containers.append(extra_body)

    for container in containers:
        if "googleSearch" in container:
            value = container.get("googleSearch")
            if value is False:
                return None
            if value is True or value is None:
                return {"googleSearch": {}}
            if isinstance(value, dict):
                return {"googleSearch": dict(value)}

        if "googleSearchRetrieval" in container:
            value = container.get("googleSearchRetrieval")
            if value is False:
                return None
            if value is True or value is None:
                return {"googleSearchRetrieval": {}}
            if isinstance(value, dict):
                return {"googleSearchRetrieval": dict(value)}

    for container in containers:
        web_search = container.get("web_search")
        if web_search is True:
            return {"googleSearch": {}}
        if web_search is False:
            return None

    return None


def _build_tool_config(tool_choice: Any) -> Optional[Dict[str, Any]]:
    if tool_choice is None:
        return None

    if isinstance(tool_choice, str):
        normalized = tool_choice.strip().lower()
        if normalized in ("", "auto"):
            return None
        if normalized == "none":
            return {"functionCallingConfig": {"mode": "NONE"}}
        if normalized in ("required", "any"):
            return {"functionCallingConfig": {"mode": "ANY"}}
        return {
            "functionCallingConfig": {
                "mode": "ANY",
                "allowedFunctionNames": [tool_choice.strip()],
            }
        }

    if isinstance(tool_choice, dict):
        function = tool_choice.get("function") or {}
        if isinstance(function, dict):
            name = str(function.get("name") or "").strip()
            if name:
                return {
                    "functionCallingConfig": {
                        "mode": "ANY",
                        "allowedFunctionNames": [name],
                    }
                }

    return None


def _build_generation_config(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    config: Dict[str, Any] = {}

    max_tokens = kwargs.get("max_tokens")
    if max_tokens is None:
        max_tokens = kwargs.get("max_completion_tokens")
    if isinstance(max_tokens, int) and max_tokens > 0:
        config["maxOutputTokens"] = max_tokens

    temperature = kwargs.get("temperature")
    if isinstance(temperature, (int, float)):
        config["temperature"] = float(temperature)

    top_p = kwargs.get("top_p")
    if isinstance(top_p, (int, float)):
        config["topP"] = float(top_p)

    return config


def _build_vertex_payload(messages: List[Dict[str, Any]], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    system_instruction, contents = _build_vertex_messages(messages)

    payload: Dict[str, Any] = {"contents": contents}
    if system_instruction:
        payload["systemInstruction"] = system_instruction

    tools = _build_vertex_tools(kwargs.get("tools"))
    native_search_tool = _build_vertex_native_search_tool(kwargs)
    if tools or native_search_tool:
        payload_tools = list(tools) if tools else []
        if native_search_tool:
            payload_tools.append(native_search_tool)
        payload["tools"] = payload_tools

    tool_config = _build_tool_config(kwargs.get("tool_choice"))
    if tool_config:
        payload["toolConfig"] = tool_config

    generation_config = _build_generation_config(kwargs)
    if generation_config:
        payload["generationConfig"] = generation_config

    return payload


def _vertex_response_to_openai(response_json: Dict[str, Any], model_name: str) -> SimpleNamespace:
    candidates = response_json.get("candidates") or []
    candidate = candidates[0] if candidates else {}
    content = candidate.get("content") if isinstance(candidate, dict) else {}
    parts = content.get("parts") if isinstance(content, dict) else []
    if not isinstance(parts, list):
        parts = []

    text, tool_calls_raw = _extract_text_and_tool_calls(parts)
    tool_calls = [
        SimpleNamespace(
            id=tool_call["id"],
            type="function",
            extra_content=tool_call.get("extra_content"),
            function=SimpleNamespace(
                name=tool_call["name"],
                arguments=tool_call["arguments"],
            ),
        )
        for tool_call in tool_calls_raw
    ]

    finish_reason = _map_finish_reason(
        candidate.get("finishReason") if isinstance(candidate, dict) else None,
        has_tool_calls=bool(tool_calls),
    )

    message = SimpleNamespace(
        role="assistant",
        content=text or None,
        tool_calls=tool_calls or None,
        reasoning_content=None,
        reasoning=None,
        reasoning_details=None,
    )
    choice = SimpleNamespace(index=0, message=message, finish_reason=finish_reason)

    return SimpleNamespace(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        object="chat.completion",
        created=int(time.time()),
        model=model_name,
        choices=[choice],
        usage=_usage_from_vertex(response_json),
    )


class VertexAIModel:
    """Low-level async REST transport for Vertex Generative AI endpoints."""

    def __init__(
        self,
        *,
        api_key: str,
        project_id: str = "",
        region: str = _DEFAULT_REGION,
        base_url: str = "",
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    ):
        self.api_key = (api_key or "").strip()
        self.project_id = (project_id or "").strip()
        self.region = _normalize_region(region)
        self.base_url = str(base_url or "").strip().rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self._client = httpx.AsyncClient()

        parsed_project, parsed_region = _parse_project_and_region_from_base_url(self.base_url)
        if not self.project_id and parsed_project:
            self.project_id = parsed_project
        if parsed_region:
            self.region = _normalize_region(parsed_region)

    @property
    def models_base_url(self) -> str:
        if self.base_url and "publishers/google/models" in self.base_url:
            return self.base_url

        computed = build_vertex_models_base_url(self.project_id, self.region)
        if computed:
            return computed

        raise ValueError(
            "Vertex endpoint is missing project/region context. "
            "Set VERTEX_PROJECT_ID and VERTEX_REGION, or set VERTEX_BASE_URL "
            "to a full '/projects/.../locations/.../publishers/google/models' URL."
        )

    async def generate_content(
        self,
        *,
        model_name: str,
        payload: Dict[str, Any],
        timeout_seconds: float,
    ) -> Dict[str, Any]:
        url = f"{self.models_base_url}/{model_name}:generateContent"
        response = await self._client.post(
            url,
            params={"key": self.api_key},
            json=payload,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    async def stream_generate_content(
        self,
        *,
        model_name: str,
        payload: Dict[str, Any],
        timeout_seconds: float,
    ) -> AsyncIterator[Dict[str, Any]]:
        url = f"{self.models_base_url}/{model_name}:streamGenerateContent"
        async with self._client.stream(
            "POST",
            url,
            params={"key": self.api_key, "alt": "sse"},
            json=payload,
            timeout=timeout_seconds,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                raw = (line or "").strip()
                if not raw:
                    continue
                if raw.startswith("data:"):
                    raw = raw[len("data:") :].strip()
                if not raw or raw == "[DONE]":
                    continue
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict):
                    yield data

    async def close(self) -> None:
        await self._client.aclose()


class VertexAIClient:
    """Sync chat.completions-compatible client backed by async httpx."""

    def __init__(
        self,
        *,
        api_key: str,
        project_id: str = "",
        region: str = _DEFAULT_REGION,
        default_model: str,
        base_url: str = "",
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    ):
        self.api_key = (api_key or "").strip()
        self.default_model = (default_model or "").strip()
        self._transport = VertexAIModel(
            api_key=self.api_key,
            project_id=project_id,
            region=region,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )
        self.base_url = self._transport.models_base_url
        self.chat = SimpleNamespace(completions=_VertexAIChatCompletions(self))
        self._closed = False

    def _resolve_model_name(self, requested: Optional[str]) -> str:
        model_name = (requested or self.default_model or "").strip()
        if model_name.startswith("models/"):
            model_name = model_name[len("models/") :]
        if not model_name:
            raise ValueError("Vertex model name is required")
        return model_name

    async def _acreate(self, **kwargs) -> SimpleNamespace:
        model_name = self._resolve_model_name(kwargs.get("model"))
        payload = _build_vertex_payload(kwargs.get("messages") or [], kwargs)
        timeout_seconds = _coerce_timeout_seconds(kwargs.get("timeout"), self._transport.timeout_seconds)
        response_json = await self._transport.generate_content(
            model_name=model_name,
            payload=payload,
            timeout_seconds=timeout_seconds,
        )
        return _vertex_response_to_openai(response_json, model_name)

    async def _acreate_stream(self, **kwargs) -> AsyncIterator[SimpleNamespace]:
        model_name = self._resolve_model_name(kwargs.get("model"))
        payload = _build_vertex_payload(kwargs.get("messages") or [], kwargs)
        timeout_seconds = _coerce_timeout_seconds(kwargs.get("timeout"), self._transport.timeout_seconds)

        prev_text = ""
        latest_tool_calls: List[Dict[str, str]] = []
        emitted_tool_name_indexes: set[int] = set()
        emitted_tool_args_indexes: set[int] = set()
        emitted_tool_extra_indexes: set[int] = set()
        last_usage: Optional[SimpleNamespace] = None

        async for response_json in self._transport.stream_generate_content(
            model_name=model_name,
            payload=payload,
            timeout_seconds=timeout_seconds,
        ):
            last_usage = _usage_from_vertex(response_json) or last_usage
            candidates = response_json.get("candidates") or []
            candidate = candidates[0] if candidates else {}
            content = candidate.get("content") if isinstance(candidate, dict) else {}
            parts = content.get("parts") if isinstance(content, dict) else []
            if not isinstance(parts, list):
                parts = []

            current_text, current_tool_calls = _extract_text_and_tool_calls(parts)
            latest_tool_calls = current_tool_calls
            delta_text = _suffix_delta(current_text, prev_text)

            finish_reason = None
            if isinstance(candidate, dict) and candidate.get("finishReason"):
                finish_reason = _map_finish_reason(
                    candidate.get("finishReason"),
                    has_tool_calls=bool(current_tool_calls),
                )

            delta_tool_calls: List[SimpleNamespace] = []
            for index, call in enumerate(current_tool_calls):
                call_id = None
                name_delta = None
                args_delta = None
                extra_delta = None

                if index not in emitted_tool_name_indexes and call.get("name"):
                    emitted_tool_name_indexes.add(index)
                    call_id = call["id"]
                    name_delta = call["name"]

                # Vertex streams functionCall args as progressively updated
                # objects, not OpenAI-style string deltas. Emit args once when
                # the turn is finishing to avoid duplicated JSON concatenation.
                if finish_reason and index not in emitted_tool_args_indexes and call.get("arguments"):
                    emitted_tool_args_indexes.add(index)
                    args_delta = call["arguments"]

                if index not in emitted_tool_extra_indexes and call.get("extra_content") is not None:
                    emitted_tool_extra_indexes.add(index)
                    extra_delta = call.get("extra_content")

                if call_id or name_delta or args_delta or extra_delta is not None:
                    delta_tool_calls.append(
                        SimpleNamespace(
                            index=index,
                            id=call_id,
                            type="function",
                            extra_content=extra_delta,
                            function=SimpleNamespace(
                                name=name_delta or None,
                                arguments=args_delta or None,
                            ),
                        )
                    )

            prev_text = current_text

            if not delta_text and not delta_tool_calls and not finish_reason:
                continue

            delta = SimpleNamespace(
                content=delta_text or None,
                tool_calls=delta_tool_calls or None,
            )
            choice = SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)
            yield SimpleNamespace(
                id=f"chatcmpl-{uuid.uuid4().hex}",
                object="chat.completion.chunk",
                created=int(time.time()),
                model=model_name,
                choices=[choice],
            )

        trailing_tool_calls: List[SimpleNamespace] = []
        for index, call in enumerate(latest_tool_calls):
            if index in emitted_tool_args_indexes:
                continue
            if not call.get("arguments"):
                continue
            call_id = call["id"] if index not in emitted_tool_name_indexes else None
            name_delta = call["name"] if index not in emitted_tool_name_indexes else None
            trailing_tool_calls.append(
                SimpleNamespace(
                    index=index,
                    id=call_id,
                    type="function",
                    extra_content=call.get("extra_content"),
                    function=SimpleNamespace(
                        name=name_delta,
                        arguments=call["arguments"],
                    ),
                )
            )

        if trailing_tool_calls:
            yield SimpleNamespace(
                id=f"chatcmpl-{uuid.uuid4().hex}",
                object="chat.completion.chunk",
                created=int(time.time()),
                model=model_name,
                choices=[
                    SimpleNamespace(
                        index=0,
                        delta=SimpleNamespace(content=None, tool_calls=trailing_tool_calls),
                        finish_reason="tool_calls",
                    )
                ],
            )

        if last_usage is not None:
            yield SimpleNamespace(
                id=f"chatcmpl-{uuid.uuid4().hex}",
                object="chat.completion.chunk",
                created=int(time.time()),
                model=model_name,
                choices=[],
                usage=last_usage,
            )

    def to_async_client(self) -> "AsyncVertexAIClient":
        return AsyncVertexAIClient(self)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        async def _close() -> None:
            await self._transport.close()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(_close())
            return

        if loop.is_running():
            # Best effort during interpreter shutdown or nested loops.
            loop.create_task(_close())
            return

        loop.run_until_complete(_close())

    def is_closed(self) -> bool:
        return self._closed


class AsyncVertexAIClient:
    """Async wrapper exposing chat.completions.create coroutine."""

    def __init__(self, sync_client: VertexAIClient):
        self._sync_client = sync_client
        self.api_key = sync_client.api_key
        self.base_url = sync_client.base_url
        self.chat = SimpleNamespace(completions=_AsyncVertexAIChatCompletions(sync_client))

    async def aclose(self) -> None:
        await self._sync_client._transport.close()

    def close(self) -> None:
        self._sync_client.close()


class _VertexAIChatCompletions:
    def __init__(self, client: VertexAIClient):
        self._client = client

    def _run_coroutine(self, coro: Any) -> Any:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        if not loop.is_running():
            return loop.run_until_complete(coro)

        # Running loop in current thread: execute the coroutine in a fresh
        # loop from a helper thread to keep sync API compatibility.
        result_box: Dict[str, Any] = {}
        error_box: Dict[str, BaseException] = {}

        def _runner() -> None:
            try:
                result_box["value"] = asyncio.run(coro)
            except BaseException as exc:  # pragma: no cover - passthrough
                error_box["error"] = exc

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join()
        if "error" in error_box:
            raise error_box["error"]
        return result_box.get("value")

    def _sync_from_async_iterator(self, factory: Any):
        item_queue: "queue.Queue[Tuple[str, Any]]" = queue.Queue()

        async def _consume() -> None:
            try:
                async for item in factory():
                    item_queue.put(("item", item))
                item_queue.put(("done", None))
            except Exception as exc:  # pragma: no cover - passthrough
                item_queue.put(("error", exc))

        def _start() -> None:
            asyncio.run(_consume())

        worker = threading.Thread(target=_start, daemon=True)
        worker.start()

        while True:
            kind, payload = item_queue.get()
            if kind == "item":
                yield payload
                continue
            if kind == "error":
                raise payload
            break

    def create(self, **kwargs) -> Any:
        if kwargs.get("stream"):
            async def _factory() -> AsyncIterator[SimpleNamespace]:
                async for chunk in self._client._acreate_stream(**kwargs):
                    yield chunk

            return self._sync_from_async_iterator(_factory)
        return self._run_coroutine(self._client._acreate(**kwargs))


class _AsyncVertexAIChatCompletions:
    def __init__(self, sync_client: VertexAIClient):
        self._sync_client = sync_client

    async def create(self, **kwargs) -> Any:
        if kwargs.get("stream"):
            return self._sync_client._acreate_stream(**kwargs)
        return await self._sync_client._acreate(**kwargs)


def build_vertex_client(
    *,
    api_key: str,
    default_model: str,
    base_url: str = "",
    project_id: str = "",
    region: str = _DEFAULT_REGION,
) -> VertexAIClient:
    """Factory used by the provider router and run_agent client creation."""
    parsed_project, parsed_region = _parse_project_and_region_from_base_url(base_url)

    resolved_project = (
        (project_id or "").strip()
        or parsed_project
        or os.getenv("VERTEX_PROJECT_ID", "").strip()
    )
    resolved_region = _normalize_region(
        (region or "").strip()
        or parsed_region
        or os.getenv("VERTEX_REGION", "").strip()
    )

    return VertexAIClient(
        api_key=api_key,
        project_id=resolved_project,
        region=resolved_region,
        base_url=base_url,
        default_model=default_model,
    )
