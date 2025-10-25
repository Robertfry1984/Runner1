#!/usr/bin/env python3
"""
Launch an OpenAI-compatible HTTP endpoint for the Gemma 3 270M Instruct model
using llama.cpp's Python server bindings.

The server binds to 127.0.0.1:54546, advertises an OpenAI-compatible `/v1`
API, keeps all execution on CPU/RAM (no GPU layers), and expands the context
window to 32,768 tokens while tuning a few knobs for better CPU throughput.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent
    model_path = project_root / "Model" / "gemma-3-270m-it-Q8_0.gguf"

    if not model_path.exists():
        print(f"[!] Model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    try:
        # Imports from llama-cpp-python's server module
        from llama_cpp.server.app import create_app
        from llama_cpp.server.settings import ModelSettings, ServerSettings
    except ImportError as exc:  # pragma: no cover - handled at runtime
        print(
            "[!] Missing dependency: llama-cpp-python with server extras.\n"
            "    Install it with: pip install \"llama-cpp-python[server]\"",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    # Make sure llama.cpp never tries to offload anything to GPU backends.
    os.environ.setdefault("LLAMA_CUBLAS", "0")
    os.environ.setdefault("LLAMA_METAL", "0")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

    cpu_count = multiprocessing.cpu_count()
    # Leave one core free for the OS if possible to improve interactivity.
    inference_threads = max(cpu_count - 1, 1)

    server_settings = ServerSettings(
        host="127.0.0.1",
        port=54546,
        # Enable request interruption so concurrent clients stay responsive.
        interrupt_requests=True,
    )

    model_settings = ModelSettings(
        model=str(model_path),
        model_alias="gemma-3-270m-it",
        chat_format="gemma",  # Gemma-specific prompt formatting
        n_ctx=32768,
        n_gpu_layers=0,  # force CPU-only execution
        n_threads=inference_threads,
        n_threads_batch=cpu_count,
        n_batch=1024,
        n_ubatch=1024,
        use_mmap=True,
        use_mlock=False,
        cache=True,
        cache_type="ram",
        cache_size=4 << 30,  # 4 GiB cache for faster reuse of prompt evals
        offload_kqv=False,
        flash_attn=False,
        logits_all=False,
        verbose=True,
    )

    app = create_app(
        server_settings=server_settings,
        model_settings=[model_settings],
    )

    # Lazy import so that we do not require uvicorn unless we actually run.
    import uvicorn  # type: ignore

    uvicorn.run(
        app,
        host=server_settings.host,
        port=server_settings.port,
        log_level="info",
        ssl_keyfile=server_settings.ssl_keyfile,
        ssl_certfile=server_settings.ssl_certfile,
    )


if __name__ == "__main__":
    main()
