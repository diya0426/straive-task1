import asyncio
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Literal

from openai import AsyncOpenAI, RateLimitError, APIStatusError
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

load_dotenv()


# Logging 


log_formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] [%(name)s] [%(process)d] [%(threadName)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


file_handler = logging.FileHandler("app.log", encoding="utf-8")
file_handler.setLevel(logging.WARNING)   # ✅ Only one setLevel call
file_handler.setFormatter(log_formatter)


console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(log_formatter)

logger = logging.getLogger("summarizer")
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(console_handler)


app = FastAPI(
    title="LLM Summarisation Service",
    version="2.0.0",
    description="Production-ready async summarisation API backed by Groq (free) or OpenAI.",
)


# Groq client 

def get_client() -> AsyncOpenAI:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set in .env file.")
    return AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )


# Schemas

SummaryStyle = Literal["brief", "detailed", "bullet_points", "eli5"]

class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Text to summarise (>=10 chars).")
    style: SummaryStyle = Field("brief", description="Summary style: brief | detailed | bullet_points | eli5")

    @field_validator("text")
    @classmethod
    def strip_text(cls, v: str) -> str:
        return v.strip()

class SummarizeResponse(BaseModel):
    request_id: str
    style: str
    summary: str
    tokens_used: int
    latency_ms: float

class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=1000)

class BatchResponse(BaseModel):
    request_id: str
    total: int
    success_count: int
    failure_count: int
    latency_ms: float
    results: list[dict]



STYLE_PROMPTS: dict[str, str] = {
    "brief": "Summarise the following text in 2-3 concise sentences.",
    "detailed": "Write a thorough, detailed summary of the following text, covering all key points.",
    "bullet_points": "Summarise the following text as a clear bullet-point list. Each bullet must be one sentence.",
    "eli5": "Explain the following text in simple language a 5-year-old could understand.",
}

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 1.5
MODEL = "llama-3.1-8b-instant"

async def call_llm(client: AsyncOpenAI, text: str, style: str) -> tuple[str, int]:
    system_prompt = STYLE_PROMPTS.get(style, STYLE_PROMPTS["brief"])
    last_exc: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = await client.chat.completions.create(
                model=MODEL,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
            )
            summary = response.choices[0].message.content.strip()
            tokens = response.usage.total_tokens
            return summary, tokens

        except RateLimitError as exc:
            wait = RETRY_BACKOFF_BASE ** attempt
            logger.warning("Rate limited – retry %d/%d in %.1fs | attempt=%d", attempt, MAX_RETRIES, wait, attempt)
            await asyncio.sleep(wait)
            last_exc = exc

        except APIStatusError as exc:
            logger.error("APIStatusError status=%s message=%s", exc.status_code, exc.message)
            last_exc = exc
            break

        except Exception as exc:
            logger.error("Unexpected LLM error: %s", exc, exc_info=True)
            last_exc = exc
            break

    raise last_exc or RuntimeError("LLM call failed after retries.")


@app.middleware("http")
async def trace_requests(request: Request, call_next):
    req_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    logger.info("-> %s %s [%s]", request.method, request.url.path, req_id)
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    logger.info("<- %s %s [%s] %.1fms", request.method, request.url.path, req_id, elapsed)
    return response

@app.post("/summarize", response_model=SummarizeResponse, status_code=200)
async def summarize(payload: SummarizeRequest):
    """Summarise a single text using Groq LLM. Fully async / non-blocking."""
    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    try:
        client = get_client()
    except RuntimeError as exc:
        logger.error("[%s] Config error: %s", request_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    try:
        summary, tokens = await call_llm(client, payload.text, payload.style)
    except Exception as exc:
        logger.error("[%s] Summarisation failed | style=%s | error=%s",
                     request_id, payload.style, exc, exc_info=True)
        raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}")

    latency = (time.perf_counter() - start) * 1000
    return SummarizeResponse(
        request_id=request_id,
        style=payload.style,
        summary=summary,
        tokens_used=tokens,
        latency_ms=round(latency, 2),
    )


# POST /batch-process

BATCH_CONCURRENCY = 10

@app.post("/batch-process", response_model=BatchResponse, status_code=200)
async def batch_process(payload: BatchRequest):
    """Process up to 1,000 texts concurrently with semaphore-controlled rate limiting."""
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    total = len(payload.texts)

    try:
        client = get_client()
    except RuntimeError as exc:
        logger.error("[%s] Config error: %s", request_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    semaphore = asyncio.Semaphore(BATCH_CONCURRENCY)

    async def process_one(index: int, text: str) -> dict:
        async with semaphore:
            try:
                summary, tokens = await call_llm(client, text, style="brief")
                return {"index": index, "status": "success", "summary": summary, "tokens_used": tokens}
            except Exception as exc:
                logger.error("[%s] Batch item %d failed: %s", request_id, index, exc)
                return {"index": index, "status": "failure", "error": str(exc)}

    tasks = [process_one(i, txt) for i, txt in enumerate(payload.texts)]
    results = await asyncio.gather(*tasks)

    success_count = sum(1 for r in results if r["status"] == "success")
    failure_count = total - success_count
    latency = (time.perf_counter() - start) * 1000

    logger.info("[%s] Batch done – total=%d success=%d failure=%d latency=%.1fms",
                request_id, total, success_count, failure_count, latency)

    return BatchResponse(
        request_id=request_id,
        total=total,
        success_count=success_count,
        failure_count=failure_count,
        latency_ms=round(latency, 2),
        results=list(results),
    )


# Health check

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}