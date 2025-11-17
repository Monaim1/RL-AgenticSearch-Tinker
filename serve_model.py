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




# from openai import OpenAI, AsyncOpenAI
# from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "Qwen/Qwen3-0.6B"
# # load the tokenizer and the model
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="auto"
# )
# ### model needs to be served via vLLM
# client = AsyncOpenAI(
#     base_url="http://localhost:8000/v1",
#     api_key="whatever"  
# )

# try:
#     response = await client.chat.completions.create(
#         model="Qwen/Qwen3-0.6B",  # Use the correct model name from vLLM
#         messages=[{"role": "user", "content": "Hello!"}],
#     )
#     print("Success!")
#     print(response.choices[0].message.content)
# except Exception as e:
#     print(f"Error: {type(e).__name__}")
#     print(f"Message: {str(e)}")