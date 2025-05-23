export CUDA_VISIBLE_DEVICES=0,1,2,3
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
--host 127.0.0.1 --port 8020 --model /capacity/userdata/models/Qwen2.5-32B-Instruct \
--served-model-name Qwen2.5-32B-Instruct --gpu-memory-utilization 0.95 --dtype bfloat16 \
--tensor-parallel-size 4 --enable-auto-tool-choice --tool-call-parser hermes

export CUDA_VISIBLE_DEVICES=4,5,6,7
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server \
--host 127.0.0.1 --port 8021 --model /capacity/userdata/models/QwQ-32B \
--served-model-name QwQ --gpu-memory-utilization 0.95 --dtype bfloat16 \
--tensor-parallel-size 4 --enable-auto-tool-choice --tool-call-parser hermes

export CUDA_VISIBLE_DEVICES=0,1
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
--host 127.0.0.1 --port 8020 --model /capacity/userdata/models/Qwen2.5-32B-Instruct \
--served-model-name Qwen2.5-32B-Instruct --gpu-memory-utilization 0.9 --dtype bfloat16 \
--tensor-parallel-size 2 --enable-auto-tool-choice --tool-call-parser hermes

export CUDA_VISIBLE_DEVICES=2,3
CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server \
--host 127.0.0.1 --port 8021 --model /capacity/userdata/models/QwQ-32B \
--served-model-name QwQ --gpu-memory-utilization 0.9 --dtype bfloat16 \
--tensor-parallel-size 2 --enable-auto-tool-choice --tool-call-parser hermes

# 启动前端streamlit服务
# nohup streamlit run client.py --server.port 8080 &
# nohup streamlit run client.py --server.port 9999 &
# nohup python3.9 -m streamlit run client.py --server.port 8080