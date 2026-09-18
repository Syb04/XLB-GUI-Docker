FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    XLB_DATA_DIR=/data HOME=/home/workbench MPLCONFIGDIR=/home/workbench/.cache/matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    git nodejs libglu1-mesa libgl1 libgomp1 libxrender1 libxcursor1 libxinerama1 libsm6 libxext6 libxft2 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ARG JAX_CUDA=0
# Keep the large CUDA runtime layer independent of application dependencies.
# Geometry-library updates then reuse it during normal image rebuilds.
RUN if [ "$JAX_CUDA" = "1" ]; then pip install --no-cache-dir 'jax[cuda12]==0.11.1'; fi
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN useradd --create-home --uid 10001 workbench \
    && mkdir -p /data /home/workbench/.cache \
    && chown -R workbench:workbench /data /home/workbench
COPY --chown=workbench:workbench workbench ./workbench
COPY --chown=workbench:workbench static ./static
USER workbench
EXPOSE 8766
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8766/api/health', timeout=3)"
CMD ["python", "-m", "workbench.server", "--host", "0.0.0.0", "--port", "8766"]
