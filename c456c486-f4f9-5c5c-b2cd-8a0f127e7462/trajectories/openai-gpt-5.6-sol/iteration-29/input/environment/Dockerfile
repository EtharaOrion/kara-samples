# CUDA runtime base so the agent can train on the H100. The previous base was
# python:3.12-slim with a CPU-only torch wheel, which made the GPU invisible to
# the agent no matter what the manifest requested.
#
# Pinned by digest, like everything else here: a floating tag would let a retag
# silently change the runtime that a frozen bundle claims to pin.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04@sha256:2fcc4280646484290cc50dce5e65f388dd04352b07cbe89a635703bd1f9aedb6

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# The agent installs its own tooling and Harbor's agent adapters bootstrap
# themselves with curl, so the base has to carry these rather than assume them.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3-pip curl git build-essential tmux ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
 && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
 && ln -sf /usr/bin/python3.11 /usr/bin/python

COPY requirements.lock /tmp/requirements.lock
RUN python -m pip install --no-cache-dir --upgrade pip \
 && python -m pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cu124 \
      -r /tmp/requirements.lock

COPY entrypoint.sh /usr/local/bin/kara-entrypoint
RUN chmod 0755 /usr/local/bin/kara-entrypoint

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/kara-entrypoint"]
