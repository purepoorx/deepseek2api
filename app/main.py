from fastapi import FastAPI, Request, HTTPException, Depends
from starlette.exceptions import ClientDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
import asyncio
import httpx
import time
import json
import os

from .config import config
from .accounts import account_manager
from .deepseek import DeepSeekClient
from .openai_adapter import convert_to_openai_stream

app = FastAPI()
http_client = httpx.AsyncClient(verify=False)

@app.on_event("startup")
async def startup_event():
    await account_manager.initialize_tokens(http_client)

@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()

# --- API Key Authentication ---
async def verify_api_key(request: Request):
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token.")
    
    token = auth_header.split(" ")[1]
    if not config.api_keys:
        print("Warning: No api_keys configured. Allowing all requests.")
        return
    if token not in config.api_keys:
        raise HTTPException(status_code=403, detail="Invalid API Key.")

# --- OpenAI Compatible Endpoints ---
@app.get("/v1/models", dependencies=[Depends(verify_api_key)])
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "deepseek-chat", "object": "model", "owned_by": "deepseek"},
            {"id": "deepseek-thinking", "object": "model", "owned_by": "deepseek"}
        ]
    }

@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(request: Request):
    try:
        request_data = await request.json()
    except Exception:
        # Handle cases where the client disconnects before sending the body
        print("Client disconnected before request body was received.")
        return JSONResponse(status_code=400, content={"error": "Client disconnected."})

    prompt = request_data.get("messages", [{}])[-1].get("content")
    model = request_data.get("model", "deepseek-chat")
    stream = request_data.get("stream", False)
    
    if "thinking_enabled" in request_data:
        thinking_enabled = request_data["thinking_enabled"]
    else:
        thinking_enabled = "thinking" in model

    account = await account_manager.get_account()
    try:
        client = DeepSeekClient(token=account.token, session=http_client)
        
        if stream:
            async def generator():
                try:
                    async for chunk in convert_to_openai_stream(client.chat_stream(prompt, thinking_enabled), model):
                        yield chunk
                except ClientDisconnect:
                    print("Client disconnected.")

            return StreamingResponse(generator(), media_type="text/event-stream")
        else:
            full_content = ""
            async for chunk in client.chat_stream(prompt, thinking_enabled):
                if chunk["type"] == "content":
                    full_content += chunk["content"]
            
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{"message": {"role": "assistant", "content": full_content}, "index": 0, "finish_reason": "stop"}]
            }
    finally:
        account_manager.release_account(account)

if __name__ == "__main__":
    import uvicorn
    print("This file is not meant to be run directly. Run with: uvicorn app.main:app --host 0.0.0.0 --port 8000")