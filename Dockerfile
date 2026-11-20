FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Asia/Shanghai

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app

COPY requirements.txt requirements-postgres.txt requirements-redis.txt requirements-docker.txt ./
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY agents ./agents
COPY bge ./bge
COPY core ./core
COPY memory ./memory
COPY migrations ./migrations
COPY optim ./optim
COPY prompts ./prompts
COPY tools ./tools
COPY app.py worker.py export_report.py ./
COPY scripts/container_streamlit.py ./scripts/container_streamlit.py

RUN mkdir -p /app/data /app/appstats /app/reports \
    && chown -R app:app /app /home/app

USER app
EXPOSE 8501

HEALTHCHECK --interval=10s --timeout=5s --start-period=150s --retries=6 \
    CMD python scripts/container_streamlit.py --healthcheck

ENTRYPOINT ["python", "scripts/container_streamlit.py"]
