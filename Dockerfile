FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DUCKDB_EXTENSION_DIRECTORY=/tmp/duckdb_extensions

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY scripts ./scripts

# Bake the httpfs extension in so cold starts don't need to download it.
RUN python -c "import duckdb, os; os.makedirs('/opt/duckdb_ext', exist_ok=True); \
c=duckdb.connect(); c.execute(\"SET extension_directory='/opt/duckdb_ext'\"); \
c.execute('INSTALL httpfs')"
ENV DUCKDB_EXTENSION_DIRECTORY=/opt/duckdb_ext

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 65"]
