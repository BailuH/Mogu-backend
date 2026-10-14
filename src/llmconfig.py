"""LLM 配置中心。

所有模型实例只在这里创建，业务代码通过 ``models["别名"]`` 或
``get_llm()`` 取用，不要在别处直接 new ChatOpenAI。

切换模型：改 ``MODEL_REGISTRY`` 里的条目，或设置环境变量
``DEFAULT_LLM`` 为某个别名（不设置则用 ``DEFAULT_MODEL_KEY``）。
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# 可复现性：全项目固定 temperature 和 seed
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED = 42

# 模型注册表：别名 -> 实例化参数。api_key / base_url 仍走 OPENAI_API_KEY / OPENAI_API_BASE 环境变量。
MODEL_REGISTRY = {
    "DeepSeek_V4": {"model": "DeepSeek-V4-Flash"},
    "Qwen_3.8": {"model": "Qwen3.8-Flash"}
}

# 不设置 DEFAULT_LLM 环境变量时使用的默认别名
DEFAULT_MODEL_KEY = "DeepSeek_V4"


def build_llm(model: str | None = None, **overrides) -> ChatOpenAI:
    """按注册表参数创建一个新的 ChatOpenAI 实例。

    model 为 None 时取默认模型；overrides 可临时覆盖 temperature、seed 等。
    """
    key = model or os.getenv("DEFAULT_LLM") or DEFAULT_MODEL_KEY
    if key not in MODEL_REGISTRY:
        raise KeyError(f"未知模型别名: {key!r}，可选: {list(MODEL_REGISTRY)}")
    params = {
        "streaming": True,
        "temperature": DEFAULT_TEMPERATURE,
        "seed": DEFAULT_SEED,
        **MODEL_REGISTRY[key],
        **overrides,
    }
    return ChatOpenAI(**params)


def get_llm(model: str | None = None, **overrides) -> ChatOpenAI:
    """取模型实例（带缓存，不传 overrides 时同一个别名只创建一次）。"""
    key = model or DEFAULT_MODEL_KEY
    if overrides or key not in _cache:
        _cache[key] = build_llm(key, **overrides)
    return _cache[key]


_cache: dict[str, ChatOpenAI] = {}

# 兼容原有导入方式：from src.llmconfig import models
models = {key: get_llm(key) for key in MODEL_REGISTRY}
