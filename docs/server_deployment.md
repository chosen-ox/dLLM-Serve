# Server Deployment Guide

This guide covers deploying dLLM-Serve as a production REST API server using FastAPI.

## Quick Start

### Basic Deployment

```bash
# Set model path
export MODEL_PATH="./LLaDA-8B-Instruct"

# Start server (single worker)
uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 1
```

The server will start on `http://0.0.0.0:8000`

## API Endpoints

### POST /v1/generate

Submit a single generation request.

**Request:**

```bash
curl -X POST "http://localhost:8000/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Tell me about AI",
    "temperature": 0.6,
    "gen_length": 128
  }'
```

**Response:**

```json
{
  "request_ids": [0],
  "status": "submitted"
}
```

### POST /v1/generate_batch

Submit multiple generation requests.

**Request:**

```bash
curl -X POST "http://localhost:8000/v1/generate_batch" \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": ["What is AI?", "Explain ML."],
    "temperature": 0.0,
    "gen_length": 64
  }'
```

**Response:**

```json
{
  "request_ids": [0, 1],
  "status": "submitted"
}
```

### GET /v1/result/{request_id}

Retrieve generation result.

**Request:**

```bash
curl "http://localhost:8000/v1/result/0"
```

**Response (Running):**

```json
{
  "request_id": 0,
  "status": "running",
  "text": null
}
```

**Response (Finished):**

```json
{
  "request_id": 0,
  "status": "finished",
  "text": "AI is artificial intelligence..."
}
```

### GET /v1/health

Health check endpoint.

**Request:**

```bash
curl "http://localhost:8000/v1/health"
```

**Response:**

```json
{
  "status": "ok"
}
```

## Request Parameters

### Generate Request

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompt` | string | Yes | - | Text prompt for generation |
| `temperature` | float | No | 0.0 | Sampling temperature |
| `gen_length` | int | No | 32 | Number of tokens to generate |

### Generate Batch Request

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompts` | list[string] | Yes | - | List of text prompts |
| `temperature` | float | No | 0.0 | Sampling temperature |
| `gen_length` | int | No | 32 | Number of tokens to generate |

**Note:** The server automatically sets `steps=gen_length` and `cfg_scale=0.0` for all requests.

## Python Client Example

```python
import requests
import time

def generate_text(prompt: str, base_url: str = "http://localhost:8000") -> str:
    """Generate text using the dLLM-Serve API."""

    # Submit request
    response = requests.post(f"{base_url}/v1/generate", json={
        "prompt": prompt,
        "temperature": 0.6,
        "gen_length": 128,
    })

    request_id = response.json()["request_ids"][0]
    print(f"Request ID: {request_id}")

    # Poll for result
    while True:
        result = requests.get(f"{base_url}/v1/result/{request_id}")
        data = result.json()

        if data["status"] == "finished":
            return data["text"]
        elif data["status"] == "running":
            time.sleep(0.1)  # Wait 100ms
        else:
            raise Exception(f"Unknown status: {data['status']}")

# Usage
output = generate_text("Tell me about machine learning")
print(output)
```

## Advanced Configuration

### Model Path

Set via environment variable:

```bash
export MODEL_PATH="/path/to/model"
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Or modify `server/app.py`:

```python
MODEL_PATH = "/path/to/model"
```

### Multiple Workers

For production, run multiple worker processes:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4
```

**Important:** Each worker loads its own model instance. Ensure sufficient GPU memory.

### GPU Configuration

For multi-GPU setups, set tensor parallelism:

```python
# In server/app.py
app.state.worker = EngineWorker(
    MODEL_PATH,
    tensor_parallel_size=2,  # Use 2 GPUs
    enforce_eager=True,
)
```

### Custom Port

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8080
```

## Production Deployment

### Using Gunicorn

For production deployment with Gunicorn:

```bash
gunicorn server.app:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 300
```

### Using Docker

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -e . --no-build-isolation

ENV MODEL_PATH="/models/LLaDA-8B-Instruct"
EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

Build and run:

```bash
docker build -t dllm-serve .
docker run -p 8000:8000 --gpus all -v /path/to/models:/models dllm-serve
```

### Using Systemd

Create `/etc/systemd/system/dllm-serve.service`:

```ini
[Unit]
Description=dLLM-Serve API Server
After=network.target

[Service]
Type=notify
User=www-data
WorkingDirectory=/path/to/dllm-serve
Environment="MODEL_PATH=/path/to/model"
Environment="CUDA_VISIBLE_DEVICES=0"
ExecStart=/usr/bin/uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 1
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable dllm-serve
sudo systemctl start dllm-serve
sudo systemctl status dllm-serve
```

### Using Nginx Reverse Proxy

Configure Nginx to proxy requests:

```nginx
upstream dllm_backend {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://dllm_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s;
        proxy_connect_timeout 300s;
    }
}
```

## Performance Tuning

### Batch Processing

Process multiple prompts efficiently using batch endpoint:

```python
prompts = ["Prompt 1", "Prompt 2", "Prompt 3", "Prompt 4", "Prompt 5"]
response = requests.post(f"{base_url}/v1/generate_batch", json={
    "prompts": prompts,
    "temperature": 0.0,
    "gen_length": 64,
})
request_ids = response.json()["request_ids"]

# Retrieve results
results = []
for rid in request_ids:
    while True:
        result = requests.get(f"{base_url}/v1/result/{rid}").json()
        if result["status"] == "finished":
            results.append(result["text"])
            break
        # Status is "running" - wait and retry
        time.sleep(0.1)
```

### Request Queuing

Implement client-side request queuing:

```python
from queue import Queue
from threading import Thread

def request_worker(queue):
    while True:
        prompt, callback = queue.get()
        try:
            result = generate_text(prompt)
            callback(result)
        except Exception as e:
            callback(None, e)
        queue.task_done()

request_queue = Queue()
for _ in range(4):  # 4 worker threads
    t = Thread(target=request_worker, args=(request_queue,))
    t.daemon = True
    t.start()

# Submit requests
request_queue.put(("Prompt 1", lambda x: print(x)))
request_queue.put(("Prompt 2", lambda x: print(x)))
```

### Monitoring

Monitor server health and performance:

```python
import time

def monitor_server(base_url: str = "http://localhost:8000"):
    """Monitor server health and request completion."""
    while True:
        # Health check
        health = requests.get(f"{base_url}/v1/health").json()
        print(f"Server status: {health['status']}")

        time.sleep(10)  # Check every 10 seconds
```

## Security Considerations

### Authentication

Add API key authentication:

```python
# In server/app.py
from fastapi import Header, HTTPException

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != "your-secret-api-key":
        raise HTTPException(status_code=403, detail="Invalid API Key")

@app.post("/v1/generate", dependencies=[Depends(verify_api_key)])
def submit_generate(req: GenerateRequest):
    ...
```

### Rate Limiting

Implement rate limiting:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/v1/generate")
@limiter.limit("10/minute")
def submit_generate(req: GenerateRequest):
    ...
```

### Input Validation

Add strict input validation:

```python
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4096)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    gen_length: int = Field(32, ge=1, le=512)
```

## Troubleshooting

### Connection Refused

Check if server is running:

```bash
curl "http://localhost:8000/v1/health"
```

### Out of Memory

Reduce batch size or enable sparse attention.

### Slow Responses

- Reduce `gen_length` parameter
- Enable sparse attention
- Increase worker count
- Use faster GPU

### Request Not Found

Ensure you're polling the correct `request_id` returned from submit.

## Next Steps

- See [API Reference](api_reference.md) for complete API documentation
- See [Installation Guide](installation.md) for setup instructions
