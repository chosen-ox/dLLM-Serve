import os
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from dllmserve.sampling_params import SamplingParams
from server.engine_worker import EngineWorker


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt")
    temperature: float = 0.0
    gen_length: int = 32


class GenerateBatchRequest(BaseModel):
    prompts: List[str]
    temperature: float = 0.0
    gen_length: int = 32


class SubmitResponse(BaseModel):
    request_ids: List[int]
    status: str = "submitted"


class ResultResponse(BaseModel):
    request_id: int
    status: str
    text: Optional[str] = None


# Use environment variable or download from HuggingFace
MODEL_PATH = os.environ.get("MODEL_PATH", "llada-8b-instruct")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create the engine worker and store it in app.state
    os.environ["DLLMSERVE_DISABLE_ATEXIT"] = "1"
    app.state.worker = EngineWorker(
        MODEL_PATH,
        tensor_parallel_size=1,
        enforce_eager=True,
    )
    yield
    app.state.worker.shutdown()


app = FastAPI(title="DLLM-Serve Server", version="0.1", lifespan=lifespan)


@app.get("/v1/health")
def health():
    return {"status": "ok"}


@app.post("/v1/generate", response_model=SubmitResponse)
def submit_generate(req: GenerateRequest):
    sp = SamplingParams(
        temperature=req.temperature,
        gen_length=req.gen_length,
        steps=req.gen_length,
        cfg_scale=0.0,
    )
    rid = app.state.worker.submit(req.prompt, sp)
    return SubmitResponse(request_ids=[rid])


@app.post("/v1/generate_batch", response_model=SubmitResponse)
def submit_generate_batch(req: GenerateBatchRequest):
    sp = SamplingParams(
        temperature=req.temperature,
        gen_length=req.gen_length,
        steps=req.gen_length,
        cfg_scale=0.0,
    )
    ids = [app.state.worker.submit(p, sp) for p in req.prompts]
    return SubmitResponse(request_ids=ids)


@app.get("/v1/result/{request_id}", response_model=ResultResponse)
def get_result(request_id: int):
    status = app.state.worker.status(request_id)
    if status == "unknown":
        raise HTTPException(status_code=404, detail="Request ID not found")
    if status == "finished":
        data = app.state.worker.get(request_id)
        return ResultResponse(
            request_id=request_id,
            status="finished",
            text=data["text"],
        )
    return ResultResponse(request_id=request_id, status=status)
