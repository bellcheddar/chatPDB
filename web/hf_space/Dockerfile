FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3 python3-pip git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# Build llama-cpp-python with CUDA support
ENV CMAKE_ARGS="-DGGML_CUDA=on"
RUN pip3 install --no-cache-dir llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

RUN pip3 install --no-cache-dir fastapi uvicorn huggingface_hub spaces

COPY app.py .

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
