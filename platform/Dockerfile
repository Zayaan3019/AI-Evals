# Build from the REPO ROOT, not this directory:
#   docker build -f platform/Dockerfile -t agentlens-platform .
# Needed because `platform` is installed as an editable dependency alongside
# `sdk` (see pyproject.toml's dependency on the sibling package via the same
# `pip install -e ./sdk -e ./platform` CI already uses in eval-gate.yml) --
# building from platform/ alone wouldn't have sdk/ in the build context.

FROM python:3.12-slim

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*

COPY sdk/ ./sdk/
COPY platform/ ./platform/

# [otlp] pulls the SDK's OTLP exporter extra -- needed by the platform's own
# ingest path tests/tooling, not by `serve` itself, but installing it here
# keeps this image capable of running the same dogfooding scripts CI uses.
RUN pip install --no-cache-dir -e ./sdk[otlp] -e ./platform

WORKDIR /build/platform

EXPOSE 8100 4317

CMD ["python", "-m", "agentlens_platform.cli", "serve"]
