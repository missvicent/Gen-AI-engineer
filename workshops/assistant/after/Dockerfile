# The capstone image — one build, two services. The compose file in
# phase8-deploy/01-compose runs this same image as the assistant API (default CMD)
# and as the MCP server (command override: python -m assistant.mcp_server --http).
#
# Multi-stage: uv resolves and installs in the first stage, the runtime stage gets
# the finished virtualenv and the source, nothing else. Both stages are bookworm so
# the venv's interpreter path survives the copy.
#
# These two bases are tags, not digests, and that is deliberate — the compose stack
# next door pins every image it PULLS by `tag@sha256:`, so the difference is worth
# stating rather than leaving to look like an oversight. A base image is the one
# place where a floating tag is the safer choice: `python:3.12-slim-bookworm` is
# rebuilt when Debian patches something in it, and a digest here would pin this
# image to a known-vulnerable OS layer until someone remembered to bump it. What
# makes the build reproducible is `uv.lock` with `--frozen`, which fixes every
# Python dependency by hash; the base supplies libc and OpenSSL, where "latest
# patch of 3.12 on bookworm" is what you actually want. Reproducibility and
# patching pull in opposite directions here, and the lockfile is what lets this
# side choose patching.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS build
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./
# `providers` alongside `integration`, because the image has to be able to run every
# brain the configuration offers. `ASSISTANT_PROVIDER=openai|anthropic` is documented
# and wired, and without this line both fail on an import inside a container that
# looks healthy — the worst shape of missing dependency, because it surfaces per
# request rather than at boot. `--no-dev` still keeps the toolchain out, and `release`
# stays out too: the reranker's ONNX runtime and RAGAS's LangChain tree are the two
# heaviest things in the repo and neither serves a request.
RUN uv sync --frozen --group integration --group providers --no-dev

FROM python:3.12-slim-bookworm
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY src/ src/
# Which commit is inside this image, baked in because there is no .git here to
# ask later. `/health` reports it, and a post-deploy smoke check compares it to
# the SHA it just shipped — the only probe that catches a half-finished rollout
# still serving an old, perfectly healthy machine.
ARG GIT_SHA=dev
ENV GIT_SHA=${GIT_SHA}
ENV PATH="/app/.venv/bin:$PATH" PYTHONPATH=/app/src
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --retries=12 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "--factory", "assistant.api:create_app", "--host", "0.0.0.0", "--port", "8000"]
