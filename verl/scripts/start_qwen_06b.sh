python -m vllm.entrypoints.openai.api_server     --model Qwen/Qwen3-Embedding-0.6B    --host 0.0.0.0     --port 8001 --quantization bitsandbytes --gpu-memory-utilization 0.22 \
 --max-model-len 4096
