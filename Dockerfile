FROM python:3.12-slim

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir mcp starlette uvicorn httpx

RUN mkdir -p /data

EXPOSE 8452

CMD ["python", "-m", "scripts.run_server", "--mode", "rest", "--state", "/data/limen.json", "--port", "8452"]
