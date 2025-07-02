ARG BASE_IMAGE=nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

# 1️⃣ basics
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install -y python3 python3-pip git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 2️⃣ python deps
COPY requirements.txt /requirements.txt
RUN pip3 install --no-cache-dir -r /requirements.txt

# 3️⃣ worker code
COPY . /app
WORKDIR /app

# 4️⃣ pick model at container-build time (or override at runtime)
ARG MODEL_ID
ENV MODEL_ID=${MODEL_ID:-stabilityai/stable-diffusion-2-1}

ENTRYPOINT ["python3", "-u", "main.py"]
