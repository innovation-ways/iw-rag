# llama-swap proxy + docker CLI for the Benchmark Studio overlay (F-00094).
#
# The upstream llama-swap image ships ONLY the Go reverse-proxy. The generator
# activation cmd in config/llama-swap.yaml shells out to `docker run ...` over
# the mounted host docker socket to bring up each vLLM candidate — which needs
# a `docker` client binary present in the container (the socket alone is not
# enough). We add the statically-linked CLI (libc-independent, so it runs on
# the glibc-based cuda base) instead of bind-mounting a host binary, so the
# overlay is self-contained and portable.
#
# Pin both the proxy tag and the CLI version; bump here + record in
# 90-meta/tech-stack-and-versions.md. Upstream pruned the bare `vNNN` tags, so
# the proxy is published only as arch-suffixed builds; the bundled CUDA
# llama.cpp is unused (we use llama-swap purely as a swap proxy).
ARG LLAMA_SWAP_TAG=v143-cuda-b5998
ARG DOCKER_CLI_VERSION=27.5.1

FROM curlimages/curl:8.11.1 AS docker-cli
ARG DOCKER_CLI_VERSION
RUN curl -fsSL "https://download.docker.com/linux/static/stable/x86_64/docker-${DOCKER_CLI_VERSION}.tgz" \
      -o /tmp/docker.tgz \
 && tar -xzf /tmp/docker.tgz -C /tmp docker/docker

FROM ghcr.io/mostlygeek/llama-swap:${LLAMA_SWAP_TAG}
COPY --from=docker-cli /tmp/docker/docker /usr/local/bin/docker
