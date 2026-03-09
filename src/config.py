import os

ENDPOINT = {'ollama_chat': 'http://localhost:11434', 'hosted_vllm': 'http://127.0.0.1:{port}/v1'}

HF_TOKEN = os.getenv('HF_TOKEN', None)