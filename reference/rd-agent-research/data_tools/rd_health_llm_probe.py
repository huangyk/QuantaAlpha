"""QuantaAlpha LLM 探针（详细中文日志，不打印密钥）。

适配 QuantaAlpha 项目：
  - 读取【仓库根目录】的 .env（D:/qqmt/QuantaAlpha/.env），而不是旧 rd-agent 的
    reference/rd-agent-research/.env
  - 用 QuantaAlpha 真正使用的【原生 openai 客户端】发起请求，调用方式与
    quantaalpha/llm/client.py 的 APIBackend.chat_client 完全一致：
        openai.OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        client.chat.completions.create(model=CHAT_MODEL, ...)
  - 键名采用 QuantaAlpha 规范：OPENAI_BASE_URL（非旧项目的 OPENAI_API_BASE）
  - 旧版本用 LiteLLM + openai/ 前缀 + OLLAMA_*，与本项目的客户端不一致，已废弃

注意：QuantaAlpha 的 APIBackend 会把 CHAT_MODEL 原样发给网关（没有 LiteLLM 的前缀语义）。
若网关不认 openai/agnes-3.0-flash 这种前缀，本探针会如实报错，此时把模型改成裸名
agnes-3.0-flash 即可。

用法（在 quantaalpha conda 环境、项目根目录下）：
  python reference/rd-agent-research/data_tools/rd_health_llm_probe.py
  python reference/rd-agent-research/data_tools/rd_health_llm_probe.py --debug
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


# 本文件位于 <项目根>/reference/rd-agent-research/data_tools/
# parents[0]=data_tools  [1]=rd-agent-research  [2]=reference  [3]=项目根
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = PROJECT_ROOT / ".env"
LOG_DIR = PROJECT_ROOT / "log"
LOG_FILE = LOG_DIR / "qa_health_llm_probe.log"


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
        "REASONING_MODEL",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "CHAT_MAX_TOKENS",
        "EMBEDDING_MODEL",
        "EMBEDDING_BASE_URL",
        "EMBEDDING_API_KEY",
    }
    for key, value in file_env.items():
        if key in prefer_file or key not in os.environ or not os.environ.get(key):
            os.environ[key] = value


def _setup_logger(debug: bool) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("qa_health_llm_probe")
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


def _probe_embedding(logger: logging.Logger, debug: bool) -> bool | None:
    """返回 True/False = 成功/失败；返回 None = 未配置，跳过。"""
    import openai

    model = os.getenv("EMBEDDING_MODEL") or ""
    base_url = os.getenv("EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL") or ""
    api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY") or "ollama"

    logger.info("======== Embedding 探测 ========")
    if not model:
        logger.info("EMBEDDING_MODEL 未设置 → 跳过（QuantaAlpha 的 embedding 为可选项，仅 RAG 用）")
        return None

    logger.info("model=%s", model)
    logger.info("api_base=%s", base_url or "(default)")
    logger.info("api_key=%s", _mask(api_key))
    logger.info("请求文本: Hello world! / 测试向量")

    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    t0 = time.perf_counter()
    try:
        resp = client.embeddings.create(model=model, input=["Hello world!", "测试向量"])
        ms = (time.perf_counter() - t0) * 1000
        data = getattr(resp, "data", None)
        n = len(data) if data is not None else 0
        dim = len(data[0].embedding) if n and getattr(data[0], "embedding", None) is not None else 0
        logger.info("结果: OK  latency=%.0fms  vectors=%s  dim=%s", ms, n, dim)
        if debug and data:
            logger.debug("首向量预览(前8维)=%s", data[0].embedding[:8])
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
    # 部分推理模型把可见文本放在 reasoning_content
    extra = getattr(message, "reasoning_content", None)
    if extra:
        return str(extra)
    return ""


def _probe_chat(logger: logging.Logger, debug: bool) -> bool:
    import openai

    model = os.getenv("CHAT_MODEL") or os.getenv("REASONING_MODEL") or ""
    base_url = os.getenv("OPENAI_BASE_URL") or ""
    api_key = os.getenv("OPENAI_API_KEY") or ""
    # Agnes 等推理模型会先烧 reasoning_tokens；探测时不要用太小的 max_tokens
    max_tokens = int(os.getenv("CHAT_MAX_TOKENS") or "2048")

    logger.info("======== Chat 探测 ========")
    if not model:
        logger.error("CHAT_MODEL 未设置（应在根 .env）")
        return False
    if not base_url:
        logger.error("OPENAI_BASE_URL 未设置（应在根 .env；注意不是旧项目的 OPENAI_API_BASE）")
        return False
    if not api_key:
        logger.error("OPENAI_API_KEY 未设置（应在根 .env）")
        return False

    # QuantaAlpha 的 APIBackend 把 CHAT_MODEL 原样发给网关（无 LiteLLM 前缀语义）
    if model.startswith("openai/"):
        logger.warning(
            "CHAT_MODEL=%r 带 openai/ 前缀。"
            "QuantaAlpha 走原生 openai 客户端，会把前缀原样发给网关。"
            "若网关报 model not found / 404，把模型改成裸名 agnes-3.0-flash 再试。",
            model,
        )
    if base_url.endswith("/chat/completions"):
        logger.warning("OPENAI_BASE_URL 以 /chat/completions 结尾，通常应为 .../v1")

    logger.info("model=%s", model)
    logger.info("api_base=%s", base_url)
    logger.info("api_key=%s", _mask(api_key))
    logger.info("max_tokens=%s", max_tokens)

    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    messages = [
        {
            "role": "system",
            "content": "You are a quant research assistant. Answer in one short Chinese sentence. No markdown.",
        },
        {
            "role": "user",
            "content": "用一句话说清：什么是 5 日动量因子？只要定义，不要举例。",
        },
    ]
    logger.info("请求 messages=%s", json.dumps(messages, ensure_ascii=False))

    token_trials = [max_tokens]
    for n in (256, 1024):
        if max_tokens < n and n not in token_trials:
            token_trials.append(n)

    last_error: Exception | None = None
    for trial_tokens in token_trials:
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=trial_tokens,
            )
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
                logger.debug("raw_model=%s finish_reason=%s", getattr(resp, "model", None), finish)

            if str(content).strip():
                logger.info("结果: OK（max_tokens=%s）", trial_tokens)
                return True

            if finish == "length":
                logger.warning("finish_reason=length 且正文为空，提高 max_tokens 重试…")
                continue

            logger.warning("Chat 返回空内容：factor 挖掘里 JSON 解析容易失败。")
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
            "建议: 1) .env 设 CHAT_MAX_TOKENS=2048 以上  2) OPENAI_BASE_URL 改为 .../v1"
            "  3) 确认 Agnes 模型是否需要关闭 reasoning / 换非推理模型"
        )
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="QuantaAlpha LLM probe (native openai client)")
    parser.add_argument("--debug", action="store_true", help="更详细的 payload / raw 日志")
    args = parser.parse_args()

    file_env = _load_dotenv(ENV_FILE)
    _apply_env(file_env)

    logger = _setup_logger(debug=args.debug)
    logger.info("QuantaAlpha LLM 探测开始  %s", datetime.now().isoformat(timespec="seconds"))
    logger.info("cwd=%s", Path.cwd())
    logger.info("project_root=%s", PROJECT_ROOT)
    logger.info("env_file=%s exists=%s", ENV_FILE, ENV_FILE.exists())
    logger.info("日志文件=%s", LOG_FILE)

    if not ENV_FILE.exists():
        logger.error(
            "根目录 .env 不存在！请先创建：cp configs/.env.example .env，并填好 OPENAI_BASE_URL / OPENAI_API_KEY / CHAT_MODEL"
        )
        return 1

    embed_res = _probe_embedding(logger, debug=args.debug)
    chat_ok = _probe_chat(logger, debug=args.debug)

    logger.info("======== 汇总 ========")
    logger.info("Embedding: %s", "OK" if embed_res is True else ("FAIL" if embed_res is False else "SKIP(未配置)"))
    logger.info("Chat:      %s", "OK" if chat_ok else "FAIL")
    logger.info("详细日志:  %s", LOG_FILE)

    if chat_ok and (embed_res is not False):
        logger.info("结论: 大模型调用正常（Chat 通过；Embedding 通过或跳过）")
        return 0
    logger.error("结论: 大模型调用异常，请打开日志排查")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
