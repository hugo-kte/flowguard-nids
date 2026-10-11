# syntax=docker/dockerfile:1
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libpcap + tcpdump: scapy uses them to compile BPF filters for live capture
RUN apt-get update \
    && apt-get install -y --no-install-recommends libpcap0.8 tcpdump \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[dashboard]"
COPY configs ./configs

# Run as an unprivileged user; live capture is opt-in (see README: --cap-add=NET_RAW).
RUN useradd --create-home --uid 10001 flowguard \
    && mkdir -p runtime artifacts reports \
    && chown -R flowguard:flowguard /app
USER flowguard

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s CMD flowguard --help > /dev/null || exit 1
ENTRYPOINT ["flowguard"]
CMD ["--help"]
