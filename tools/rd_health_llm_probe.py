"""RD-Agent LLM probe with detailed Chinese logs (no secrets).

Usage (via conda rd-agent env):
  python tools/rd_health_llm_probe.py
  python tools/rd_health_llm_probe.py --debug
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "log"
LOG_FILE = LOG_DIR / "rd_health_llm_probe.log"
ENV_FILE = ROOT / ".env"


def _load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _mask(value: str | None, keep: int = 4) -> str:
    if not value:
        return "(empty)"
    if len(value) <= keep * 2:
        return "***"
    return f"{value[:keep]}...{value[-keep:]} (len={len(value)})"


def _apply_env(file_env: dict[str, str]) -> None:
    # .env 覆盖同名空值；关键 LLM 键以文件为准，避免壳里残留旧值干扰探测
    prefer_file = {
        "CHAT_MODEL",
        "OPENAI_API_BASE",
        "OPENAI_API_KEY",
        "CHAT_MAX_TOKENS",
        "EMBEDDING_MODEL",
        "OLLAMA_API_BASE",
        "LITELLM_PROXY_API_BASE",
        "LITELLM_PROXY_API_KEY",
    }
    for key, value in file_env.items():
        if key in prefer_file or key not in os.environ or not os.environ.get(key):
            os.environ[key] = value


def _setup_logger(debug: bool) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("rd_health_llm_probe")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8", mode="w")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _probe_embedding(logger: logging.Logger, debug: bool) -> bool:
    from litellm import embedding

    model = os.getenv("EMBEDDING_MODEL") or os.getenv("LITELLM_EMBEDDING_MODEL") or ""
    api_base = os.getenv("OLLAMA_API_BASE") or os.getenv("LITELLM_PROXY_API_BASE") or ""
    api_key = os.getenv("LITELLM_PROXY_API_KEY") or os.getenv("OPENAI_API_KEY") or "ollama"

    logger.info("======== Embedding 探测 ========")
    logger.info("model=%s", model)
    logger.info("api_base=%s", api_base or "(default)")
    logger.info("api_key=%s", _mask(api_key))
    logger.info("请求文本: Hello world! / 测试向量")

    kwargs: dict = {
        "model": model,
        "input": ["Hello world!", "测试向量"],
        "api_key": api_key,
    }
    if api_base:
        kwargs["api_base"] = api_base

    t0 = time.perf_counter()
    try:
        resp = embedding(**kwargs)
        ms = (time.perf_counter() - t0) * 1000
        data = getattr(resp, "data", None) or resp.get("data")  # type: ignore[union-attr]
        n = len(data) if data is not None else 0
        dim = len(data[0]["embedding"]) if n and "embedding" in data[0] else 0
        logger.info("结果: OK  latency=%.0fms  vectors=%s  dim=%s", ms, n, dim)
        if debug and data:
            preview = data[0]["embedding"][:8]
            logger.debug("首向量预览(前8维)=%s", preview)
        return True
    except Exception as exc:  # noqa: BLE001
        ms = (time.perf_counter() - t0) * 1000
        logger.error("结果: FAIL  latency=%.0fms  error=%s", ms, exc)
        logger.error("traceback:\n%s", traceback.format_exc())
        return False


def _message_content(message) -> str:
    content = getattr(message, "content", None)
    if content:
        return str(content)
    # Some reasoning models put visible text elsewhere
    for key in ("reasoning_content", "reasoning", "thinking"):
        extra = getattr(message, key, None)
        if extra:
            return str(extra)
    return ""


def _probe_chat(logger: logging.Logger, debug: bool) -> bool:
    from litellm import completion

    model = os.getenv("CHAT_MODEL") or ""
    api_base = os.getenv("OPENAI_API_BASE") or os.getenv("CHAT_OPENAI_BASE_URL") or ""
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("CHAT_OPENAI_API_KEY") or ""
    # Agnes 等推理模型会先烧 reasoning_tokens；探测时不要用太小的 max_tokens
    max_tokens = int(os.getenv("CHAT_MAX_TOKENS") or os.getenv("LITELLM_CHAT_MAX_TOKENS") or "512")

    # 严格按 .env 原样测，不做自动改写（避免日志和配置对不上）
    if model and "/" not in model:
        logger.warning(
            "CHAT_MODEL=%r 没有 provider 前缀。"
            "LiteLLM 通常需要写成 openai/agnes-2.5-flash（openai/ 只表示协议，不是 OpenAI 官方模型）。"
            "本探测将按你写的原样调用，若失败就是前缀问题。",
            model,
        )

    logger.info("======== Chat 探测 ========")
    logger.info("model=%s", model)
    logger.info("api_base=%s", api_base or "(default)")
    logger.info("api_key=%s", _mask(api_key))
    logger.info("max_tokens=%s", max_tokens)
    if api_base.endswith("/chat/completions"):
        logger.warning(
            "OPENAI_API_BASE 以 /chat/completions 结尾。"
            "LiteLLM 通常期望 .../v1；建议改成 https://apihub.agnes-ai.cn/v1"
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a quant research assistant. "
                "Answer in one short Chinese sentence. No markdown."
            ),
        },
        {
            "role": "user",
            "content": (
                "用一句话说清：什么是 5 日动量因子？"
                "只要定义，不要举例。"
            ),
        },
    ]
    logger.info("请求 messages=%s", json.dumps(messages, ensure_ascii=False))

    token_trials = [max_tokens]
    if max_tokens < 256:
        token_trials.append(256)
    if max_tokens < 1024:
        token_trials.append(1024)
    # unique preserve order
    seen: set[int] = set()
    trials = []
    for n in token_trials:
        if n not in seen:
            seen.add(n)
            trials.append(n)

    last_error: Exception | None = None
    for trial_tokens in trials:
        kwargs: dict = {
            "model": model,
            "messages": messages,
            "api_key": api_key,
            "temperature": 0,
            "max_tokens": trial_tokens,
        }
        if api_base:
            kwargs["api_base"] = api_base

        t0 = time.perf_counter()
        try:
            resp = completion(**kwargs)
            ms = (time.perf_counter() - t0) * 1000
            choice0 = resp.choices[0] if resp.choices else None
            message = choice0.message if choice0 else None
            content = _message_content(message) if message else ""
            finish = getattr(choice0, "finish_reason", None) if choice0 else None
            usage = getattr(resp, "usage", None)

            logger.info("尝试 max_tokens=%s → latency=%.0fms finish_reason=%s", trial_tokens, ms, finish)
            logger.info("回复内容: %r", content)
            if usage is not None:
                logger.info("usage=%s", usage)
                details = getattr(usage, "completion_tokens_details", None)
                reasoning_tokens = getattr(details, "reasoning_tokens", None) if details else None
                text_tokens = getattr(details, "text_tokens", None) if details else None
                if reasoning_tokens and (not text_tokens) and (not str(content).strip()):
                    logger.warning(
                        "推理模型把输出额度几乎全用在 reasoning_tokens=%s，正文 text_tokens=%s。"
                        "这不是 Key 失效，而是 max_tokens 太小或模型只思考不回答。",
                        reasoning_tokens,
                        text_tokens,
                    )
            if debug and choice0 is not None:
                logger.debug(
                    "raw_model=%s finish_reason=%s",
                    getattr(resp, "model", None),
                    finish,
                )

            if str(content).strip():
                logger.info("结果: OK（max_tokens=%s）", trial_tokens)
                return True

            if finish == "length":
                logger.warning("finish_reason=length 且正文为空，提高 max_tokens 重试…")
                continue

            logger.warning("Chat 返回空内容：fin_factor 里 JSON 解析容易失败。")
        except Exception as exc:  # noqa: BLE001
            ms = (time.perf_counter() - t0) * 1000
            last_error = exc
            logger.error("尝试 max_tokens=%s FAIL  latency=%.0fms  error=%s", trial_tokens, ms, exc)
            logger.error("traceback:\n%s", traceback.format_exc())

    if last_error is not None:
        logger.error("结果: FAIL  最后异常: %s", last_error)
    else:
        logger.error(
            "结果: FAIL  Chat 多次仍为空。"
            "建议: 1) .env 设 CHAT_MAX_TOKENS=2048  2) OPENAI_API_BASE 改为 .../v1"
            "  3) 确认 Agnes 模型是否需要关闭 reasoning / 换非推理模型"
        )
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe Chat/Embedding with detailed logs")
    parser.add_argument("--debug", action="store_true", help="More verbose LiteLLM + payload logs")
    args = parser.parse_args()

    file_env = _load_dotenv(ENV_FILE)
    _apply_env(file_env)

    if args.debug:
        os.environ.setdefault("LITELLM_LOG", "DEBUG")
        os.environ.setdefault("LOG_LLM_CHAT_CONTENT", "true")
    else:
        os.environ.setdefault("LITELLM_LOG", "INFO")

    logger = _setup_logger(debug=args.debug)
    logger.info("RD-Agent LLM 探测开始  %s", datetime.now().isoformat(timespec="seconds"))
    logger.info("cwd=%s", Path.cwd())
    logger.info("env_file=%s exists=%s", ENV_FILE, ENV_FILE.exists())
    logger.info("日志文件=%s", LOG_FILE)

    embed_ok = _probe_embedding(logger, debug=args.debug)
    chat_ok = _probe_chat(logger, debug=args.debug)

    logger.info("======== 汇总 ========")
    logger.info("Embedding: %s", "OK" if embed_ok else "FAIL")
    logger.info("Chat:      %s", "OK" if chat_ok else "FAIL")
    logger.info("详细日志:  %s", LOG_FILE)

    if embed_ok and chat_ok:
        logger.info("结论: 大模型调用正常")
        return 0
    logger.error("结论: 大模型调用异常，请打开日志排查")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
