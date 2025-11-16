#!/usr/bin/env python3
"""
Simple vLLM server for Qwen3-0.6B
Run with: uv run serve_model.py
"""
import subprocess
import sys

if __name__ == "__main__":
    # Use subprocess to call vllm serve directly
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", "Qwen/Qwen3-0.6B",
        "--dtype", "auto",
        "--max-model-len", "2048",
        "--port", "8000",
        "--trust-remote-code",
        "--disable-log-requests",  # Reduce logging verbosity
    ]
    print("Starting vLLM server on http://localhost:8000")
    print("OpenAI-compatible endpoint: http://localhost:8000/v1")
    subprocess.run(cmd)
