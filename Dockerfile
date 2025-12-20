FROM --platform=linux/arm64 ghcr.io/astral-sh/uv:debian

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     ca-certificates \
#     && rm -rf /var/lib/apt/lists/*

# COPY requirements.txt /workspace/requirements.txt
# RUN python3 -m pip install --break-system-packages -r /workspace/requirements.txt

WORKDIR /workspace
COPY . /workspace
