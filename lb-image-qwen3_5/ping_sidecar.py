"""Minimal /ping shim for RunPod load-balancer.

RunPod's load-balancing endpoint health-checks GET /ping on PORT_HEALTH.
Convention:
    200 -> healthy (worker takes traffic)
    204 -> initializing (RunPod waits)
    other -> unhealthy (worker is killed)

vLLM's OpenAI server exposes /health (not /ping). We proxy.
"""
import os
import httpx
import uvicorn
from fastapi import FastAPI, Response

VLLM_HEALTH_URL = "http://127.0.0.1:8000/health"

app = FastAPI()


@app.get("/ping")
async def ping():
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(VLLM_HEALTH_URL)
            if r.status_code == 200:
                return {"status": "healthy"}
    except Exception:
        # vLLM not yet listening — say "still initializing".
        pass
    return Response(status_code=204)


if __name__ == "__main__":
    port = int(os.environ.get("PORT_HEALTH", "5000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
