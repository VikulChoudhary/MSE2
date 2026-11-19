FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 10001 helios
COPY . .
RUN mkdir -p /app/runtime && chown -R helios:helios /app/runtime
USER helios
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,json,sys; d=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3)); sys.exit(0 if d.get('status')=='healthy' else 1)" || exit 1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
