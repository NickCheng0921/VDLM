"""
VDLM API Server - OpenAI-compatible FastAPI implementation for MDM diffusion LM.
"""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any
import time
import uuid

import logging
import logging.handlers
import multiprocessing

try:
    multiprocessing.set_start_method(
        "spawn", force=True
    )  # pytest complains about fork usage
except RuntimeError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s %(processName)s(pid=%(process)d) [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%m-%d %H:%M:%S",
)
logger = logging.getLogger("VDLM_API")

from llm_engine import LLMEngine

engine = LLMEngine()
pending_requests: Dict[str, asyncio.Future] = {}


async def response_watcher():
    """
    Background task that waits for results from the engine process.
    Uses run_in_executor to block on the queue without blocking the asyncio loop.
    """
    logger.info("Response watcher started.")
    loop = asyncio.get_running_loop()
    while True:
        try:
            # Run the blocking 'get' in a thread pool so the main loop keeps running
            result = await loop.run_in_executor(None, engine.response_queue.get)
            if result is None:
                break

            request_id, text = result
            logger.debug(f"Response watcher received result for {request_id}")

            if request_id in pending_requests:
                future = pending_requests.pop(request_id)
                if not future.done():  # TODO, when will this future be set elsewhere?
                    future.set_result(text)
            else:
                logger.warning(
                    f"Request ID {request_id} not found in pending_requests!"
                )
        except Exception as e:
            logger.error(f"Error in response watcher: {e}")
            await asyncio.sleep(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting LLM Engine...")
    engine.start()
    watcher_task = asyncio.create_task(response_watcher())
    yield
    logger.info("Stopping LLM Engine...")
    engine.stop()

    # Unblock the response_watcher by sending a poison pill to the response queue
    engine.response_queue.put(None)

    try:
        await asyncio.wait_for(watcher_task, timeout=2.0)
    except asyncio.TimeoutError:
        logger.warning("Response watcher task did not exit in time, cancelling...")
        watcher_task.cancel()


app = FastAPI(title="VDLM API Server", lifespan=lifespan)


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    logprobs: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None


class CompletionChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionChoice]
    usage: CompletionUsage


@app.get("/health")
async def health_check():
    """Health check endpoint to verify if the model is loaded."""
    if not engine.ready_event.is_set():
        raise HTTPException(status_code=503, detail="Model is loading")
    return {"status": "ok", "model_loaded": True}


@app.post("/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest):
    """
    Endpoint for OpenAI-style completions using a separate LLM Engine process.
    """
    if not engine.ready_event.is_set():
        raise HTTPException(
            status_code=503, detail="Model is still loading. Please try again later."
        )

    request_id = str(uuid.uuid4())
    future = asyncio.get_running_loop().create_future()
    pending_requests[request_id] = future

    # Extract single prompt if it's a list for this simple skeleton
    prompt = request.prompt if isinstance(request.prompt, str) else request.prompt[0]

    engine.submit_request(request_id, prompt)

    try:
        generated_text = await asyncio.wait_for(future, timeout=30.0)
    except asyncio.TimeoutError:
        pending_requests.pop(request_id, None)
        raise HTTPException(status_code=504, detail="LLM Engine timeout")

    choices = [CompletionChoice(text=generated_text, index=0, finish_reason="stop")]

    usage = CompletionUsage(
        prompt_tokens=len(prompt.split()),  # Very rough estimate
        completion_tokens=len(generated_text.split()),
        total_tokens=len(prompt.split()) + len(generated_text.split()),
    )

    return CompletionResponse(model=request.model, choices=choices, usage=usage)


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="VDLM API Server")
    parser.add_argument(
        "--mock", action="store_true", help="Run with mock LLM engine (fast mode)"
    )
    args = parser.parse_args()

    if args.mock:
        logger.info("Starting in MOCK mode")
        engine = LLMEngine(is_mock=True)

    uvicorn.run(app, host="0.0.0.0", port=8000)
